# Persistent read-only Windows CPU clock. Compile before benchmark sampling.
$ErrorActionPreference = 'Stop'
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class BlazeHostClock {
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern bool GetSystemTimes(out long idle, out long kernel, out long user);
}
'@
while ($null -ne [Console]::ReadLine()) {
    [long]$idle = 0; [long]$kernel = 0; [long]$user = 0
    if (-not [BlazeHostClock]::GetSystemTimes([ref]$idle, [ref]$kernel, [ref]$user)) {
        throw 'GetSystemTimes failed'
    }
    $process = [Diagnostics.Process]::GetCurrentProcess()
    @{
        busy_seconds = ($kernel + $user - $idle) / 10000000.0
        idle_seconds = $idle / 10000000.0
        collector_cpu_seconds = $process.TotalProcessorTime.TotalSeconds
        logical_processors = [Environment]::ProcessorCount
        utc = [DateTime]::UtcNow.ToString('o')
    } | ConvertTo-Json -Compress
    [Console]::Out.Flush()
}
