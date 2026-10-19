// Radiance - SKSE plugin.
// Loads the baked probe volume (.rdp) of the interior cell the player is in, uploads it as six RGBA16F 3D
// textures and a small settings buffer, and binds them to five free pixel shader slots (ShaderSlots in Radiance.ini) every frame. The
// patched Community Shaders Lighting.hlsl (written by the baker) reads them in place of the flat ambient.

#include "PCH.h"

using Microsoft::WRL::ComPtr;

namespace
{
	// Pixel shader slots: atlas R, G, B, settings, brick indirection (see Radiance.hlsli). The baker picks slots no
	// shader of the installed Community Shaders variant uses and writes them to Radiance.ini (ShaderSlots).
	constexpr UINT kSlotCount = 5;
	constexpr UINT kParamSlot = 3;
	constexpr UINT kIndirectionSlot = 4;
	std::array<UINT, kSlotCount> g_slots{ 110, 111, 112, 116, 117 };

	std::string SlotList()
	{
		return fmt::format("t{}, t{}, t{}, t{}, t{}", g_slots[0], g_slots[1], g_slots[2], g_slots[3], g_slots[4]);
	}

	// Defaults match the bake: brightness is calibrated per cell at bake time, so 1.0 / 0.1 is "as baked".
	struct Settings
	{
		bool  enabled = true;
		float bounceGain = 1.0f;    // multiplier on the baked light
		float floorMult = 1.0f;     // multiplier on each cell's baked share of its original flat ambient
		float dayWeight = 1.0f;     // daylight / static layer weight by day
		float nightWeight = 1.0f;   // ... and at night (1 = as the lighting mod placed it)
		float normalOffset = 24.0f; // sample this far in front of the surface (the bake assumes 24)
		int   toggleKey = VK_F7;
		int   debugKey = VK_F8;     // cycles the debug view: off / magenta where active / baked light only
	} g_settings;

	struct alignas(16) ProbeParams
	{
		float originInvSpacing[4];
		float dims[4];
		float weights[4];
		float bricks[4];
		float atlas[4];
	};
	static_assert(sizeof(ProbeParams) == 80);

	struct Volume
	{
		std::string                                     key;  // "skyrim.esm/01605E"
		bool                                            valid = false;
		float                                           origin[3]{};
		float                                           spacing = 1.0f;
		std::uint32_t                                   dims[3]{};
		std::uint32_t                                   brick = 4;
		std::uint32_t                                   bricks[3]{};
		std::uint32_t                                   atlasBricks[3]{};
		float                                           floorShare = 0.1f;  // share of the original ambient kept
		std::array<ComPtr<ID3D11ShaderResourceView>, 6> srv;
		std::array<ComPtr<ID3D11Texture3D>, 3>          combined;   // fixtures + daylight x weight, per channel
		std::array<std::vector<std::uint16_t>, 6>       cpu;        // both layers, to recombine when the weight changes
		float                                           dayApplied = -1.0f;
		std::uint32_t                                   atlasSize[3]{};
		ComPtr<ID3D11ShaderResourceView>                indirection;
	} g_volume;

	ID3D11Device*                    g_device = nullptr;
	ID3D11DeviceContext*             g_context = nullptr;
	ComPtr<ID3D11Buffer>             g_paramBuffer;
	ComPtr<ID3D11ShaderResourceView> g_paramSRV;
	bool                             g_toggledOn = true;
	bool                             g_keyWasDown = false;
	bool                             g_debugKeyWasDown = false;
	int                              g_debugView = 0;
	std::string                      g_lastCellKey;

	// what we want bound in our slots; re-applied whenever someone else clears them
	std::array<ID3D11ShaderResourceView*, kSlotCount> g_views{};
	thread_local bool                                 t_inOwnBind = false;
	std::uint64_t                                     g_foreignWrites = 0;
	std::uint64_t                                     g_clearStates = 0;
	std::uint64_t                                     g_foreignReal = 0;
	std::uint64_t                                     g_framesLost = 0;
	std::uint64_t                                     g_frames = 0;

	using PresentFn = HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain*, UINT, UINT);
	PresentFn g_originalPresent = nullptr;
	using PSSetSRVFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, UINT, ID3D11ShaderResourceView* const*);
	PSSetSRVFn g_originalPSSetSRV = nullptr;
	using ClearStateFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*);
	ClearStateFn g_originalClearState = nullptr;

	void BindOwn(ID3D11DeviceContext* ctx)
	{
		t_inOwnBind = true;
		for (UINT i = 0; i < kSlotCount; ++i) {
			g_originalPSSetSRV(ctx, g_slots[i], 1, &g_views[i]);
		}
		t_inOwnBind = false;
	}

	std::string CallerModule(void* address)
	{
		HMODULE mod = nullptr;
		char    name[MAX_PATH] = "?";
		if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
				static_cast<LPCSTR>(address), &mod)) {
			GetModuleFileNameA(mod, name, MAX_PATH);
		}
		return std::filesystem::path(name).filename().string();
	}

	void STDMETHODCALLTYPE HookedPSSetShaderResources(ID3D11DeviceContext* ctx, UINT start, UINT count,
		ID3D11ShaderResourceView* const* views)
	{
		g_originalPSSetSRV(ctx, start, count, views);
		if (t_inOwnBind || !g_views[kParamSlot]) {
			return;
		}
		bool touched = false;
		bool real = false;
		for (UINT i = 0; i < kSlotCount; ++i) {
			if (g_slots[i] >= start && g_slots[i] < start + count) {
				touched = true;
				// someone else's data in our slot (not our own view put back, e.g. by an overlay restoring state)
				auto* v = views ? views[g_slots[i] - start] : nullptr;
				real |= v != nullptr && v != g_views[i];
			}
		}
		if (!touched) {
			return;
		}
		++g_foreignWrites;
		// never overwrite real data (it belongs to their draw); log who
		if (real) {
			++g_foreignReal;
			static int logged = 0;
			if (logged < 20) {
				++logged;
				spdlog::warn("slots t{}-t{} were given real data by {} - not overwriting", start, start + count - 1,
					CallerModule(_ReturnAddress()));
			}
			return;
		}
		static int loggedClear = 0;
		if (loggedClear < 3) {
			++loggedClear;
			spdlog::info("slots t{}-t{} cleared by {} - restoring ours", start, start + count - 1,
				CallerModule(_ReturnAddress()));
		}
		BindOwn(ctx);  // a plain clear: restore ours
	}

	void STDMETHODCALLTYPE HookedClearState(ID3D11DeviceContext* ctx)
	{
		g_originalClearState(ctx);
		if (g_views[kParamSlot]) {
			++g_clearStates;
			BindOwn(ctx);
		}
	}

	template <class T>
	T HookVTable(void* object, std::size_t index, T replacement)
	{
		auto** vtable = *reinterpret_cast<void***>(object);
		DWORD  old;
		VirtualProtect(&vtable[index], sizeof(void*), PAGE_EXECUTE_READWRITE, &old);
		auto original = reinterpret_cast<T>(vtable[index]);
		vtable[index] = reinterpret_cast<void*>(replacement);
		VirtualProtect(&vtable[index], sizeof(void*), old, &old);
		return original;
	}

	// ---------------------------------------------------------------------------------------------
	void LoadSettings()
	{
		const char* path = "Data\\SKSE\\Plugins\\Radiance.ini";
		char        buf[64];
		auto        getf = [&](const char* key, float def) {
            char d[32];
            std::snprintf(d, sizeof(d), "%g", def);
            GetPrivateProfileStringA("Settings", key, d, buf, sizeof(buf), path);
            return static_cast<float>(std::atof(buf));
		};
		g_settings.enabled = GetPrivateProfileIntA("Settings", "Enabled", 1, path) != 0;
		g_settings.bounceGain = getf("BounceGain", g_settings.bounceGain);
		g_settings.floorMult = getf("AmbientFloorMultiplier", g_settings.floorMult);
		g_settings.dayWeight = getf("DaylightDay", g_settings.dayWeight);
		g_settings.nightWeight = getf("DaylightNight", g_settings.nightWeight);
		g_settings.normalOffset = getf("NormalOffset", g_settings.normalOffset);
		g_settings.toggleKey = GetPrivateProfileIntA("Settings", "ToggleKey", VK_F7, path);
		g_settings.debugKey = GetPrivateProfileIntA("Settings", "DebugKey", VK_F8, path);
		char slots[128];
		GetPrivateProfileStringA("Settings", "ShaderSlots", "", slots, sizeof(slots), path);
		std::array<UINT, kSlotCount> parsed{};
		int                          n = 0;
		for (const char* c = slots; *c && n < static_cast<int>(kSlotCount);) {
			char* end = nullptr;
			const long v = std::strtol(c, &end, 10);
			if (end == c) {
				++c;
				continue;
			}
			if (v >= 0 && v < D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT) {
				parsed[n++] = static_cast<UINT>(v);
			}
			c = end;
		}
		if (n == static_cast<int>(kSlotCount)) {
			g_slots = parsed;
		} else if (slots[0]) {
			spdlog::error("ShaderSlots in Radiance.ini is invalid ({}) - using {}", slots, SlotList());
		}
		spdlog::info("settings: enabled={} gain={} floor={} day={} night={} offset={} keys=0x{:X}/0x{:X} slots {}",
			g_settings.enabled, g_settings.bounceGain, g_settings.floorMult, g_settings.dayWeight,
			g_settings.nightWeight, g_settings.normalOffset, g_settings.toggleKey, g_settings.debugKey, SlotList());
	}

	// ---------------------------------------------------------------------------------------------
	// .rdp reader (format documented in radiance/probefile.py)
	bool LoadVolume(const std::filesystem::path& path, Volume& out)
	{
		std::ifstream f(path, std::ios::binary);
		if (!f) {
			return false;
		}
		std::vector<char> d((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
		std::size_t       o = 0;
		auto              need = [&](std::size_t n) { return o + n <= d.size(); };
		auto              rd = [&](void* dst, std::size_t n) { std::memcpy(dst, d.data() + o, n); o += n; };
		if (!need(8) || std::memcmp(d.data(), "RDNC", 4) != 0) {
			spdlog::warn("{}: not a probe file", path.string());
			return false;
		}
		o = 4;
		std::uint32_t version;
		rd(&version, 4);
		if (version != 2) {
			spdlog::warn("{}: probe file version {} - rebake this cell with the current baker", path.string(), version);
			return false;
		}
		std::uint16_t nameLen;
		rd(&nameLen, 2);
		o += nameLen + 4;  // plugin name, local id
		if (!need(12 + 4 + 12 + 8 + 20 + 8 + 4 + 12 + 12 + 12 + 16)) {
			return false;
		}
		rd(out.origin, 12);
		rd(&out.spacing, 4);
		rd(out.dims, 12);
		std::uint32_t layers, flags, nbricks;
		float         calibration;
		rd(&layers, 4);
		rd(&flags, 4);
		o += 20 + 8;  // fingerprint, rays/bounces
		rd(&out.brick, 4);
		rd(out.bricks, 12);
		rd(out.atlasBricks, 12);
		rd(&nbricks, 4);
		rd(&calibration, 4);
		rd(&out.floorShare, 4);
		std::uint64_t rawSize, packedSize;
		rd(&rawSize, 8);
		rd(&packedSize, 8);
		// payload: zlib-compressed indirection + atlas
		std::vector<char> payload;
		if (flags & 2) {
			if (!need(packedSize) || rawSize > (1ull << 32)) {
				spdlog::warn("{}: truncated probe file", path.string());
				return false;
			}
			payload.resize(rawSize);
			uLongf outLen = static_cast<uLongf>(rawSize);
			if (uncompress(reinterpret_cast<Bytef*>(payload.data()), &outLen,
					reinterpret_cast<const Bytef*>(d.data() + o), static_cast<uLong>(packedSize)) != Z_OK ||
				outLen != rawSize) {
				spdlog::warn("{}: corrupt probe file", path.string());
				return false;
			}
		} else {
			payload.assign(d.begin() + o, d.end());
		}
		d.swap(payload);
		o = 0;
		const std::uint32_t pb = out.brick + 1;
		const std::size_t   cells = std::size_t(out.bricks[0]) * out.bricks[1] * out.bricks[2];
		const std::uint32_t aw = out.atlasBricks[0] * pb, ah = out.atlasBricks[1] * pb, ad = out.atlasBricks[2] * pb;
		const std::size_t   block = std::size_t(aw) * ah * ad * 4 * sizeof(std::uint16_t);
		if (layers < 2 || cells == 0 || nbricks == 0 || aw > 2048 || ah > 2048 || ad > 2048 ||
			!need(cells * 4 + block * 6)) {
			spdlog::warn("{}: truncated or unexpected layout", path.string());
			return false;
		}
		// brick indirection: R32_UINT, brick id + 1 (0 = no data)
		{
			D3D11_TEXTURE3D_DESC desc{};
			desc.Width = out.bricks[0];
			desc.Height = out.bricks[1];
			desc.Depth = out.bricks[2];
			desc.MipLevels = 1;
			desc.Format = DXGI_FORMAT_R32_UINT;
			desc.Usage = D3D11_USAGE_IMMUTABLE;
			desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
			D3D11_SUBRESOURCE_DATA init{ d.data() + o, out.bricks[0] * 4u, out.bricks[0] * out.bricks[1] * 4u };
			ComPtr<ID3D11Texture3D> tex;
			if (FAILED(g_device->CreateTexture3D(&desc, &init, tex.GetAddressOf())) ||
				FAILED(g_device->CreateShaderResourceView(tex.Get(), nullptr, out.indirection.ReleaseAndGetAddressOf()))) {
				spdlog::error("{}: could not create the brick indirection texture", path.string());
				return false;
			}
			o += cells * 4;
		}
		out.atlasSize[0] = aw;
		out.atlasSize[1] = ah;
		out.atlasSize[2] = ad;
		const std::size_t halves = block / sizeof(std::uint16_t);
		for (int i = 0; i < 6; ++i) {
			out.cpu[i].resize(halves);
			std::memcpy(out.cpu[i].data(), d.data() + o + block * i, block);
		}
		// one texture per color channel holding fixtures + daylight (weighted): the shader samples 3, not 6
		for (int i = 0; i < 3; ++i) {
			D3D11_TEXTURE3D_DESC desc{};
			desc.Width = aw;
			desc.Height = ah;
			desc.Depth = ad;
			desc.MipLevels = 1;
			desc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
			desc.Usage = D3D11_USAGE_DEFAULT;
			desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
			D3D11_SUBRESOURCE_DATA init{ out.cpu[i].data(), aw * 8, aw * ah * 8 };
			if (FAILED(g_device->CreateTexture3D(&desc, &init, out.combined[i].GetAddressOf())) ||
				FAILED(g_device->CreateShaderResourceView(out.combined[i].Get(), nullptr,
					out.srv[i].ReleaseAndGetAddressOf()))) {
				spdlog::error("{}: could not create probe atlas texture {}", path.string(), i);
				return false;
			}
		}
		spdlog::info("  {} bricks, atlas {}x{}x{}, bounce x{:.2f}, original ambient kept {:.0f}%", nbricks, aw, ah, ad,
			calibration, out.floorShare * 100.0f);
		out.valid = true;
		return true;
	}

	// fixtures layer + daylight layer x weight -> the three combined textures (only when the weight changed)
	void CombineLayers(Volume& v, float dayWeight)
	{
		if (!v.valid || std::abs(v.dayApplied - dayWeight) < 0.02f) {
			return;
		}
		v.dayApplied = dayWeight;
		const std::size_t      n = v.cpu[0].size();
		std::vector<std::uint16_t> tmp(n);
		for (int c = 0; c < 3; ++c) {
			const auto& fix = v.cpu[c];
			const auto& day = v.cpu[3 + c];
			for (std::size_t j = 0; j < n; ++j) {
				const float f = DirectX::PackedVector::XMConvertHalfToFloat(fix[j]) +
				                dayWeight * DirectX::PackedVector::XMConvertHalfToFloat(day[j]);
				tmp[j] = DirectX::PackedVector::XMConvertFloatToHalf(f);
			}
			g_context->UpdateSubresource(v.combined[c].Get(), 0, nullptr, tmp.data(), v.atlasSize[0] * 8,
				v.atlasSize[0] * v.atlasSize[1] * 8);
		}
	}

	bool CreateParamBuffer()
	{
		D3D11_BUFFER_DESC desc{};
		desc.ByteWidth = sizeof(ProbeParams);
		desc.Usage = D3D11_USAGE_DEFAULT;
		desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
		desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
		desc.StructureByteStride = sizeof(ProbeParams);
		ProbeParams zero{};
		D3D11_SUBRESOURCE_DATA init{ &zero, 0, 0 };
		if (FAILED(g_device->CreateBuffer(&desc, &init, g_paramBuffer.GetAddressOf()))) {
			return false;
		}
		D3D11_SHADER_RESOURCE_VIEW_DESC sd{};
		sd.Format = DXGI_FORMAT_UNKNOWN;
		sd.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
		sd.Buffer.FirstElement = 0;
		sd.Buffer.NumElements = 1;
		return SUCCEEDED(g_device->CreateShaderResourceView(g_paramBuffer.Get(), &sd, g_paramSRV.GetAddressOf()));
	}

	// ---------------------------------------------------------------------------------------------
	// load-order independent form key, as the baker writes it: "skyrim.esm/01605E" (defining plugin / local id)
	std::string FormKey(const RE::TESForm* form)
	{
		auto* file = form->GetFile(0);
		if (!file) {
			return {};
		}
		std::string plugin(file->GetFilename());
		for (auto& c : plugin) {
			c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
		}
		const auto id = form->GetFormID() & (file->IsLight() ? 0xFFFu : 0xFFFFFFu);
		return plugin + "/" + fmt::format("{:06X}", id);
	}

	std::string CellKey(RE::TESObjectCELL* cell, std::filesystem::path& path)
	{
		auto key = FormKey(cell);
		if (key.empty()) {
			return {};
		}
		const auto slash = key.find('/');
		path = std::filesystem::path("Data") / "Radiance" / "Probes" / key.substr(0, slash) /
		       (key.substr(slash + 1) + ".rdp");
		return key;
	}

	// ---------------------------------------------------------------------------------------------
	// Lights the bake replaced (fill / fully baked, Radiance\lights.txt). They stay in the game data
	// untouched; while baked lighting is on they are hidden (Community Shaders skips hidden lights, so they
	// cost nothing) and the toggle key shows them again, so it compares against the original setup. Only
	// the loaded 3D light is changed: nothing is written into the save game.
	std::unordered_map<std::string, std::vector<std::string>> g_cellLights;  // cell key -> light ref keys
	struct HiddenLight
	{
		RE::NiPointer<RE::NiLight> light;
		RE::NiColor                diffuse;
		float                      fade;
		bool                       wasCulled;
	};
	std::vector<HiddenLight> g_hidden;
	std::vector<RE::FormID>  g_lightRefs;  // references in the current cell to hide
	std::string              g_lightsCell;
	std::uint32_t            g_lightsScan = 0;
	std::atomic<bool>        g_lightTaskPending = false;

	void LoadLightList()
	{
		std::ifstream f("Data\\Radiance\\lights.txt");
		std::string   line, cell;
		std::size_t   n = 0;
		while (std::getline(f, line)) {
			while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) {
				line.pop_back();
			}
			if (line.empty() || line[0] == '#') {
				continue;
			}
			if (line.rfind("cell ", 0) == 0) {
				cell = line.substr(5);
			} else if (!cell.empty()) {
				g_cellLights[cell].push_back(line);
				++n;
			}
		}
		spdlog::info("{} replaced lights in {} baked cells (hidden while baked lighting is on)", n, g_cellLights.size());
	}

	RE::NiLight* RefLight(RE::TESObjectREFR* ref)
	{
		auto* x = ref->extraList.GetByType<RE::ExtraLight>();
		return x && x->lightData ? x->lightData->light.get() : nullptr;
	}

	void RestoreLights()
	{
		for (auto& h : g_hidden) {
			auto& rd = h.light->GetLightRuntimeData();
			rd.diffuse = h.diffuse;
			rd.fade = h.fade;
			h.light->SetAppCulled(h.wasCulled);
		}
		g_hidden.clear();
	}

	// main thread (SKSE task)
	void UpdateLights(const std::string& cellKey, bool hide)
	{
		auto* player = RE::PlayerCharacter::GetSingleton();
		auto* cell = player ? player->GetParentCell() : nullptr;
		std::string key = cell && cell->IsInteriorCell() ? FormKey(cell) : std::string();
		if (key != cellKey) {
			return;  // changed cell since the frame that queued this: next frame
		}
		if (key != g_lightsCell || ++g_lightsScan % 120 == 0) {
			// (re)find the references: on a new cell, and every 2 s for references that load late
			const bool newCell = key != g_lightsCell;
			if (newCell) {
				RestoreLights();
				g_lightsCell = key;
			}
			g_lightRefs.clear();
			auto it = g_cellLights.find(key);
			if (cell && it != g_cellLights.end()) {
				const auto& want = it->second;
				cell->ForEachReference([&](RE::TESObjectREFR* ref) {
					auto* base = ref ? ref->GetBaseObject() : nullptr;
					if (base && base->GetFormType() == RE::FormType::Light &&
						std::find(want.begin(), want.end(), FormKey(ref)) != want.end()) {
						g_lightRefs.push_back(ref->GetFormID());
					}
					return RE::BSContainer::ForEachResult::kContinue;
				});
				if (newCell) {
					spdlog::info("cell {}: {} of {} replaced lights found", key, g_lightRefs.size(), want.size());
				}
			}
		}
		if (!hide) {
			RestoreLights();
			return;
		}
		for (auto id : g_lightRefs) {
			auto* ref = RE::TESForm::LookupByID<RE::TESObjectREFR>(id);
			auto* light = ref ? RefLight(ref) : nullptr;
			if (!light) {
				continue;
			}
			auto& rd = light->GetLightRuntimeData();
			auto  h = std::find_if(g_hidden.begin(), g_hidden.end(), [&](auto& e) { return e.light.get() == light; });
			if (h == g_hidden.end()) {
				g_hidden.push_back({ RE::NiPointer<RE::NiLight>(light), rd.diffuse, rd.fade,
					light->GetFlags().any(RE::NiAVObject::Flag::kHidden) });
			}
			// every frame: the engine may rewrite fade (flicker) or culling
			light->SetAppCulled(true);
			rd.fade = 0.0f;
			rd.diffuse = RE::NiColor(0.0f, 0.0f, 0.0f);
		}
	}

	float DaylightWeight()
	{
		auto* cal = RE::Calendar::GetSingleton();
		const float h = cal ? cal->GetHour() : 12.0f;
		auto smooth = [](float e0, float e1, float x) {
			float t = std::clamp((x - e0) / (e1 - e0), 0.0f, 1.0f);
			return t * t * (3.0f - 2.0f * t);
		};
		const float day = smooth(5.0f, 7.5f, h) * (1.0f - smooth(18.5f, 21.0f, h));
		return g_settings.nightWeight + (g_settings.dayWeight - g_settings.nightWeight) * day;
	}

	void UpdateAndBind()
	{
		if (!g_context || !g_paramSRV) {
			return;
		}
		// toggle key (edge triggered)
		const bool down = (GetAsyncKeyState(g_settings.toggleKey) & 0x8000) != 0;
		if (down && !g_keyWasDown) {
			g_toggledOn = !g_toggledOn;
			spdlog::info("toggled {}", g_toggledOn ? "on" : "off");
			RE::SendHUDMessage::ShowHUDMessage(g_toggledOn ? "Radiance: ON" : "Radiance: OFF");
		}
		g_keyWasDown = down;
		const bool debugDown = (GetAsyncKeyState(g_settings.debugKey) & 0x8000) != 0;
		if (debugDown && !g_debugKeyWasDown) {
			g_debugView = (g_debugView + 1) % 3;
			static const char* names[] = { "Radiance debug: off", "Radiance debug: MAGENTA where active",
				"Radiance debug: probes only" };
			spdlog::info("{}", names[g_debugView]);
			RE::SendHUDMessage::ShowHUDMessage(names[g_debugView]);
		}
		g_debugKeyWasDown = debugDown;

		// was our binding still in place at the end of the frame? (diagnostic)
		++g_frames;
		if (g_views[kParamSlot]) {
			ID3D11ShaderResourceView* cur = nullptr;
			g_context->PSGetShaderResources(g_slots[kParamSlot], 1, &cur);
			if (cur != g_views[kParamSlot]) {
				++g_framesLost;
			}
			if (cur) {
				cur->Release();
			}
		}
		if (g_frames % 900 == 0 && g_volume.valid) {
			spdlog::info("binding check: {} frames, slots lost at frame end {}x, foreign writes to our slots {}x "
						 "({} with real data, left alone), ClearState {}x", g_frames, g_framesLost, g_foreignWrites,
				g_foreignReal, g_clearStates);
		}

		// current cell -> probe volume
		RE::TESObjectCELL* cell = nullptr;
		if (auto* player = RE::PlayerCharacter::GetSingleton()) {
			cell = player->GetParentCell();
		}
		std::filesystem::path path;
		std::string           key;
		if (cell && cell->IsInteriorCell()) {
			key = CellKey(cell, path);
		}
		if (key != g_lastCellKey) {
			g_lastCellKey = key;
			g_volume = Volume{};
			if (!key.empty()) {
				g_volume.key = key;
				if (LoadVolume(path, g_volume)) {
					spdlog::info("cell {}: loaded {} ({}x{}x{} probes @ {})", key, path.string(), g_volume.dims[0],
						g_volume.dims[1], g_volume.dims[2], g_volume.spacing);
				} else {
					spdlog::info("cell {}: no bake ({})", key, path.string());
				}
			}
		}

		ProbeParams p{};
		const bool  on = g_settings.enabled && g_toggledOn && g_volume.valid;
		if (on) {
			p.originInvSpacing[0] = g_volume.origin[0];
			p.originInvSpacing[1] = g_volume.origin[1];
			p.originInvSpacing[2] = g_volume.origin[2];
			p.originInvSpacing[3] = 1.0f / g_volume.spacing;
			p.dims[0] = float(g_volume.dims[0]);
			p.dims[1] = float(g_volume.dims[1]);
			p.dims[2] = float(g_volume.dims[2]);
			p.dims[3] = 1.0f;
			CombineLayers(g_volume, DaylightWeight());
			p.weights[0] = g_settings.bounceGain;
			p.weights[1] = 0.0f;  // daylight already combined into the fixture textures
			p.weights[2] = g_volume.floorShare * g_settings.floorMult;
			p.weights[3] = g_settings.normalOffset;
			p.bricks[0] = float(g_volume.bricks[0]);
			p.bricks[1] = float(g_volume.bricks[1]);
			p.bricks[2] = float(g_volume.bricks[2]);
			p.bricks[3] = float(g_volume.brick);
			p.atlas[0] = float(g_volume.atlasBricks[0]);
			p.atlas[1] = float(g_volume.atlasBricks[1]);
			p.atlas[2] = float(g_volume.atlasBricks[2]);
			p.atlas[3] = float(g_debugView);
		}
		g_context->UpdateSubresource(g_paramBuffer.Get(), 0, nullptr, &p, 0, 0);

		for (int i = 0; i < 3; ++i) {
			g_views[i] = on ? g_volume.srv[i].Get() : nullptr;
		}
		g_views[kParamSlot] = g_paramSRV.Get();
		g_views[kIndirectionSlot] = on ? g_volume.indirection.Get() : nullptr;
		BindOwn(g_context);

		// replaced lights: hidden while the bake shows, back on when toggled off (on the main thread)
		if (!g_cellLights.empty() && !g_lightTaskPending.exchange(true)) {
			SKSE::GetTaskInterface()->AddTask([key, on]() {
				UpdateLights(key, on);
				g_lightTaskPending = false;
			});
		}
	}

	HRESULT STDMETHODCALLTYPE HookedPresent(IDXGISwapChain* swapChain, UINT syncInterval, UINT flags)
	{
		const HRESULT hr = g_originalPresent(swapChain, syncInterval, flags);
		// bound after Present: stays bound for the whole next frame (our hooks restore it if anything clears our slots)
		UpdateAndBind();
		return hr;
	}

	void InstallRendererHooks()
	{
		auto* renderer = RE::BSGraphics::Renderer::GetSingleton();
		if (!renderer) {
			spdlog::error("renderer not found");
			return;
		}
		auto& data = renderer->GetRuntimeData();
		g_device = reinterpret_cast<ID3D11Device*>(data.forwarder);
		g_context = reinterpret_cast<ID3D11DeviceContext*>(data.context);
		auto* swapChain = reinterpret_cast<IDXGISwapChain*>(data.renderWindows[0].swapChain);
		if (!g_device || !g_context || !swapChain) {
			spdlog::error("D3D11 objects not available (device {} context {} swapchain {})", (void*)g_device,
				(void*)g_context, (void*)swapChain);
			return;
		}
		if (!CreateParamBuffer()) {
			spdlog::error("could not create the settings buffer");
			return;
		}
		// vtable hooks (chain with other hooks): IDXGISwapChain::Present (8), and on the immediate context
		// PSSetShaderResources (8) + ClearState (110) so our slots are restored if anything overwrites them
		g_originalPSSetSRV = HookVTable(g_context, 8, &HookedPSSetShaderResources);
		g_originalClearState = HookVTable(g_context, 110, &HookedClearState);
		g_originalPresent = HookVTable(swapChain, 8, &HookedPresent);
		spdlog::info("hooked Present, PSSetShaderResources, ClearState; shader slots {}", SlotList());
	}

	void InitLog()
	{
		auto dir = SKSE::log::log_directory();
		if (!dir) {
			return;
		}
		auto sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>((*dir / "Radiance.log").string(), true);
		auto log = std::make_shared<spdlog::logger>("log", std::move(sink));
		log->set_level(spdlog::level::info);
		log->flush_on(spdlog::level::info);
		spdlog::set_default_logger(std::move(log));
		spdlog::set_pattern("[%H:%M:%S] %v");
	}
}

SKSEPluginLoad(const SKSE::LoadInterface* skse)
{
	SKSE::Init(skse);
	InitLog();
	spdlog::info("Radiance {} loaded ({})", SKSE::PluginDeclaration::GetSingleton()->GetVersion().string(),
		REL::Module::IsVR() ? "VR" : "SE/AE");
	LoadSettings();
	SKSE::GetMessagingInterface()->RegisterListener([](SKSE::MessagingInterface::Message* msg) {
		if (msg->type == SKSE::MessagingInterface::kDataLoaded) {
			LoadLightList();
			InstallRendererHooks();
		}
	});
	return true;
}

// Offline check used by the baker's tests: load a probe file with a private D3D11 device, exactly like the game.
// Returns the size of the brick grid, 0 when the file is rejected, -1 when no device could be created.
extern "C" __declspec(dllexport) int Radiance_SelfTest(const char* path)
{
	ComPtr<ID3D11Device>        dev;
	ComPtr<ID3D11DeviceContext> ctx;
	if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION,
			dev.GetAddressOf(), nullptr, ctx.GetAddressOf()))) {
		return -1;
	}
	auto* saved = g_device;
	g_device = dev.Get();
	Volume v;
	const bool ok = LoadVolume(path, v);
	g_device = saved;
	return ok ? static_cast<int>(v.bricks[0] * v.bricks[1] * v.bricks[2]) : 0;
}

// Offline check of the slot binding with a private D3D11 device: our resources must land in the configured slots
// and come back after the things the game and overlays do (ClearState, clearing slots, binding over a range).
// Returns 1 when all checks pass, else the number of the first failing check (negative), -100 without a device.
extern "C" __declspec(dllexport) int Radiance_BindTest(const int* slots)
{
	ComPtr<ID3D11Device>        dev;
	ComPtr<ID3D11DeviceContext> ctx;
	if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION,
			dev.GetAddressOf(), nullptr, ctx.GetAddressOf()))) {
		return -100;
	}
	for (UINT i = 0; i < kSlotCount; ++i) {
		g_slots[i] = static_cast<UINT>(slots[i]);
	}
	g_device = dev.Get();
	if (!CreateParamBuffer()) {
		return -101;
	}
	g_originalPSSetSRV = HookVTable(ctx.Get(), 8, &HookedPSSetShaderResources);
	g_originalClearState = HookVTable(ctx.Get(), 110, &HookedClearState);
	g_views = {};
	g_views[kParamSlot] = g_paramSRV.Get();
	auto bound = [&]() {
		ID3D11ShaderResourceView* cur = nullptr;
		ctx->PSGetShaderResources(g_slots[kParamSlot], 1, &cur);
		const bool ok = cur == g_paramSRV.Get();
		if (cur) {
			cur->Release();
		}
		return ok;
	};
	int result = 1;
	BindOwn(ctx.Get());
	if (!bound()) {
		result = -1;
	}
	ctx->ClearState();
	if (result == 1 && !bound()) {
		result = -2;
	}
	ID3D11ShaderResourceView* none = nullptr;
	ctx->PSSetShaderResources(g_slots[kParamSlot], 1, &none);
	if (result == 1 && !bound()) {
		result = -3;
	}
	std::array<ID3D11ShaderResourceView*, D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT> all{};
	ctx->PSSetShaderResources(0, D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT, all.data());
	if (result == 1 && !bound()) {
		result = -4;
	}
	// a slot next to ours (another feature's resource) must be left alone
	const UINT other = g_slots[kParamSlot] > 0 ? g_slots[kParamSlot] - 1 : g_slots[kParamSlot] + 1;
	bool otherIsOurs = false;
	for (auto s : g_slots) {
		otherIsOurs |= s == other;
	}
	if (!otherIsOurs) {
		ctx->PSSetShaderResources(other, 1, &none);
		ID3D11ShaderResourceView* cur = nullptr;
		ctx->PSGetShaderResources(other, 1, &cur);
		if (result == 1 && cur != nullptr) {
			result = -5;
		}
		if (cur) {
			cur->Release();
		}
	}
	HookVTable(ctx.Get(), 8, g_originalPSSetSRV);
	HookVTable(ctx.Get(), 110, g_originalClearState);
	g_views = {};
	g_paramSRV.Reset();
	g_paramBuffer.Reset();
	g_device = nullptr;
	return result;
}
