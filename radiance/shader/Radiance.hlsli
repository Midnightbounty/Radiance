// Radiance - baked indirect light (probe volume) for interior cells.
// Installed by the Radiance baker next to a patched Lighting.hlsl. The SKSE plugin Radiance.dll
// fills these resources for the current cell; when it doesn't (exteriors, cells without a bake, plugin
// missing or toggled off) Dims.w is 0 and the game's own ambient is used unchanged.
//
// The volume is sparse: the probe grid is split into bricks of Bricks.w cells; an indirection texture holds
// brick id + 1 (0 = no data) and the atlas stores each brick with a shared border, so one hardware-filtered
// fetch per texture is enough.
//
// The t-registers (@SLOT_...@) are filled in by the baker with texture slots that no shader of the installed
// Community Shaders variant uses; Radiance.ini (ShaderSlots) tells the SKSE plugin the same slots.
#ifndef RADIANCE_HLSLI
#define RADIANCE_HLSLI

namespace Radiance
{
	struct ProbeParams
	{
		float4 OriginInvSpacing;  // xyz: world position of probe (0,0,0); w: 1 / spacing
		float4 Dims;              // xyz: probe grid size; w: 1 = enabled
		float4 Weights;           // x: baked light weight; y: unused; z: share of original ambient kept; w: normal offset
		float4 Bricks;            // xyz: brick grid size; w: cells per brick edge
		float4 Atlas;             // xyz: atlas size in bricks; w: debug view (0 off, 1 magenta where active, 2 baked light only)
	};

	// atlas, one RGBA16F volume per color channel (all layers pre-combined by the plugin): (a, bx, by, bz),
	// irradiance(n) = a + b.n
	Texture3D<float4> BLFixR : register(t@SLOT_R@);
	Texture3D<float4> BLFixG : register(t@SLOT_G@);
	Texture3D<float4> BLFixB : register(t@SLOT_B@);
	StructuredBuffer<ProbeParams> BLParams : register(t@SLOT_PARAMS@);
	Texture3D<uint> BLIndirection : register(t@SLOT_INDIRECTION@);

	float3 Ambient(float3 vanillaAmbient, float3 positionWS, float3 normalWS, SamplerState samp)
	{
		ProbeParams p = BLParams[0];
		[branch] if (p.Dims.w < 0.5)
			return vanillaAmbient;

		float3 f = (positionWS + normalWS * p.Weights.w - p.OriginInvSpacing.xyz) * p.OriginInvSpacing.w;
		f = clamp(f, 0.0, p.Dims.xyz - 1.0);
		const float bs = p.Bricks.w;
		int3 b = min(int3(floor(f / bs)), int3(p.Bricks.xyz) - 1);
		uint id = BLIndirection.Load(int4(b, 0));
		[branch] if (id == 0)
			return vanillaAmbient;
		id -= 1;
		const uint ax = uint(p.Atlas.x);
		const uint ay = uint(p.Atlas.y);
		const float3 cell = float3(id % ax, (id / ax) % ay, id / (ax * ay));
		const float pb = bs + 1.0;
		float3 uvw = (cell * pb + (f - float3(b) * bs) + 0.5) / (p.Atlas.xyz * pb);

		float4 basis = float4(1.0, normalWS);
		float3 fix = float3(dot(BLFixR.SampleLevel(samp, uvw, 0), basis),
			dot(BLFixG.SampleLevel(samp, uvw, 0), basis),
			dot(BLFixB.SampleLevel(samp, uvw, 0), basis));
		float3 baked = max(fix, 0.0) * p.Weights.x;
		[branch] if (p.Atlas.w > 1.5)
			return baked;
		[branch] if (p.Atlas.w > 0.5)
			return float3(4.0, 0.0, 4.0);
		return baked + max(vanillaAmbient, 0.0) * p.Weights.z;
	}
}

#endif
