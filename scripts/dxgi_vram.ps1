# Reports the DXGI video-memory picture (the API family Steam's in-game VRAM overlay
# belongs to): per-adapter physical VRAM capacity + the DXGI per-process
# Budget/CurrentUsage memory-manager figures, via IDXGIAdapter3::QueryVideoMemoryInfo.
#
# IMPORTANT semantics (why this never equals Steam's overlay):
#   * CurrentUsage / Budget are PER-PROCESS figures returned by the driver for the
#     CALLING process. A standalone script process (this one) has booked no GPU
#     memory, so CurrentUsage is ~0 even when the system is using VRAM.
#   * Steam's overlay attaches to the GAME process and reads the GAME's D3D11/D3D12
#     device usage (its reserved/committed resources). That number (e.g. 26.6GB) is
#     private to the game process and is not reconstructible from a system utility.
#   * The per-adapter DedicatedVideoMemory below (= physical VRAM, e.g. 24GB) is the
#     denominator Steam shows (23.9GB); the numerator comes from in-process bookkeeping.
#
# This script is built on raw COM vtable calls (no [ComImport]) for reliability:
#   CreateDXGIFactory1 -> IDXGIFactory1::EnumAdapters1 (vtable[12])
#   IDXGIAdapter3::QueryVideoMemoryInfo (vtable[14]) called directly on the returned
#   object's own vtable -- no QueryInterface needed, because DXGI returns objects with
#   the full adapter vtable.
# COM vtable methods need the object pointer ('this') as their first argument; COM
# calls return HRESULTs; vtable slots are read via the object's vtable pointer
# (dereference offset 0, then index * IntPtr.Size).

$ErrorActionPreference = 'Stop'

$cs = @"
using System;
using System.Text;
using System.Runtime.InteropServices;

public static class DxgiVram {
  [StructLayout(LayoutKind.Sequential)]
  public struct M { public ulong Budget, CurrentUsage, AvailableForReservation, CurrentReservation; }

  [UnmanagedFunctionPointer(CallingConvention.StdCall)]
  delegate int EnumAdaptersFn(IntPtr This, uint Adapter, out IntPtr ppAdapter);
  [UnmanagedFunctionPointer(CallingConvention.StdCall)]
  delegate int GetDescFn(IntPtr This, IntPtr pDesc);
  [UnmanagedFunctionPointer(CallingConvention.StdCall)]
  delegate int QueryVmiFn(IntPtr This, uint NodeIndex, int MemoryType, out M info);

  [DllImport("dxgi.dll")]
  static extern int CreateDXGIFactory1(ref Guid riid, out IntPtr factory);

  static readonly Guid IID_IDXGIFactory1 = new Guid("770aae78-f26f-4dba-a829-253c83d1b387");

  static IntPtr Slot(IntPtr obj, int n) { return Marshal.ReadIntPtr(Marshal.ReadIntPtr(obj), n * IntPtr.Size); }

  public static string Report() {
    var sb = new StringBuilder();
    IntPtr factory;
    Guid fid = IID_IDXGIFactory1;
    int hr = CreateDXGIFactory1(ref fid, out factory);
    if (hr != 0) return "CreateDXGIFactory1 failed: 0x" + hr.ToString("X8");
    const double GB = 1073741824.0;
    try {
      var enumAdapters = (EnumAdaptersFn)Marshal.GetDelegateForFunctionPointer(Slot(factory, 12), typeof(EnumAdaptersFn));
      for (uint idx = 0; idx < 8; idx++) {
        IntPtr adapter;
        hr = enumAdapters(factory, idx, out adapter);
        if (hr != 0) continue; // DXGI_ERROR_NOT_FOUND

        // IDXGIAdapter::GetDesc (vtable[8]) -> physical VRAM capacity
        IntPtr buf = Marshal.AllocHGlobal(304);
        string name = "?";
        double cap = 0;
        try {
          var gd = (GetDescFn)Marshal.GetDelegateForFunctionPointer(Slot(adapter, 8), typeof(GetDescFn));
          if (gd(adapter, buf) == 0) {
            name = Marshal.PtrToStringUni(buf, 128).TrimEnd('\0');
            uint vid = (uint)Marshal.ReadInt32(buf, 256);
            cap = Marshal.ReadInt64(buf, 272) / GB;
            if (vid == 0x1414) name = "Microsoft Basic Render Driver"; // software adapter
          }
        } finally { Marshal.FreeHGlobal(buf); }

        // IDXGIAdapter3::QueryVideoMemoryInfo (vtable[14]) -- dedicated(0) & cpu-accessible(1)
        var lines = new StringBuilder();
        try {
          var qvmi = (QueryVmiFn)Marshal.GetDelegateForFunctionPointer(Slot(adapter, 14), typeof(QueryVmiFn));
          string[] ty = { "DEDICATED     ", "CPU_ACCESSIBLE" };
          int[] types = { 0, 1 };
          for (int i = 0; i < types.Length; i++) {
            M m;
            int v = qvmi(adapter, 0, types[i], out m);
            if (v == 0)
              lines.AppendFormat("      [{0}] Budget={1,7:N2}GB  CurrentUsage={2,6:N2}GB  AvailRes={3,6:N2}GB  CurRes={4,6:N2}GB{5}",
                ty[i], m.Budget/GB, m.CurrentUsage/GB, m.AvailableForReservation/GB, m.CurrentReservation/GB, Environment.NewLine);
          }
        } catch (Exception ex) { lines.AppendLine("      QueryVideoMemoryInfo unavailable: " + ex.Message); }

        sb.AppendFormat("{0}{1}  physical VRAM capacity: {2:N2} GB{3}{4}",
          Environment.NewLine, name, cap, Environment.NewLine, lines);
        Marshal.Release(adapter);
      }
    } finally { Marshal.Release(factory); }
    return sb.ToString();
  }
}
"@

Add-Type -TypeDefinition $cs -Language CSharp
Write-Output "DXGI video memory (per adapter)"
Write-Output "Note: CurrentUsage/Budget are PER-PROCESS figures for THIS (script) process."
[DxgiVram]::Report()