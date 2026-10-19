# Builds Radiance.dll (Skyrim SE/AE/VR). Dependencies and build output live in a short, neutral path:
# vcpkg and MSVC break on Windows' 260-character path limit, and source paths end up inside the DLL
# (error messages), so a path without your user name keeps it out of the release.
param(
	[string]$BuildRoot = "C:\rb",
	[string]$Config = "Release"
)
$ErrorActionPreference = 'Stop'
$src = $PSScriptRoot
$vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -prerelease -latest -property installationPath
& "$vs\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -SkipAutomaticLocation | Out-Null
$clib = Join-Path $BuildRoot 'CommonLibVR'
if (-not (Test-Path $clib)) {
	git -c core.longpaths=true clone --depth 1 --recurse-submodules --shallow-submodules https://github.com/alandtse/CommonLibVR.git $clib
}
$out = Join-Path $BuildRoot 'out'
$vcpkg = if ($env:VCPKG_ROOT) { $env:VCPKG_ROOT } else { "$vs\VC\vcpkg" }
cmake -S $src -B $out -G Ninja "-DCMAKE_BUILD_TYPE=$Config" "-DCOMMONLIB_DIR=$clib" `
	"-DCMAKE_TOOLCHAIN_FILE=$vcpkg\scripts\buildsystems\vcpkg.cmake" "-DVCPKG_TARGET_TRIPLET=x64-windows-static" `
	"-DVCPKG_INSTALLED_DIR=$BuildRoot\vcpkg_installed"
if ($LASTEXITCODE) { exit $LASTEXITCODE }
cmake --build $out
exit $LASTEXITCODE
