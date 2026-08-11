[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$ZipPath = "baseline_submit.zip",

    [string]$DataDir = "data",
    [int]$TimeoutSeconds = 600,
    [string]$ImageName = "lg-aimers9-checker:local",
    [switch]$Gpu,
    [switch]$NoBuild
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot

function Resolve-CheckerPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$InputPath,
        [Parameter(Mandatory = $true)]
        [string]$Label
    )

    $candidates = [System.Collections.Generic.List[string]]::new()
    $candidates.Add($InputPath)

    # Resolve ordinary relative paths from the project root as well, so the
    # script behaves consistently even when invoked from another directory.
    if (-not [System.IO.Path]::IsPathRooted($InputPath)) {
        $candidates.Add((Join-Path $projectRoot $InputPath))
    }

    # A common PowerShell typo is `/file.zip`. If it is only a filename (not a
    # real absolute directory path), also interpret it as a project-root file.
    if ($InputPath -match '^[\\/][^\\/]+$') {
        $candidates.Add((Join-Path $projectRoot $InputPath.TrimStart('\', '/')))
    }

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return (Resolve-Path -LiteralPath $candidate -ErrorAction Stop).Path
        }
    }

    if ($Label -eq "ZIP") {
        $leafName = [System.IO.Path]::GetFileName($InputPath)
        if ($leafName) {
            $nameMatches = @(Get-ChildItem -LiteralPath $projectRoot -Recurse -File -Filter $leafName)
            if ($nameMatches.Count -eq 1) {
                Write-Host "ZIP resolved by filename: $($nameMatches[0].FullName)"
                return $nameMatches[0].FullName
            }
            if ($nameMatches.Count -gt 1) {
                $ambiguous = $nameMatches | ForEach-Object {
                    $_.FullName.Substring($projectRoot.Length + 1)
                }
                throw "Multiple ZIP files have the name '$leafName'. Specify one path:`n  - $($ambiguous -join "`n  - ")"
            }
        }

        $available = Get-ChildItem -LiteralPath $projectRoot -Recurse -File -Filter "*.zip" |
            ForEach-Object { $_.FullName.Substring($projectRoot.Length + 1) }
        $hint = if ($available) {
            "`nAvailable ZIP files:`n  - " + ($available -join "`n  - ")
        } else {
            "`nNo ZIP files were found under the project root."
        }
        throw "ZIP file not found: $InputPath$hint"
    }

    throw "$Label path not found: $InputPath"
}

$zip = Resolve-CheckerPath -InputPath $ZipPath -Label "ZIP"
$data = Resolve-CheckerPath -InputPath $DataDir -Label "DataDir"

if ([System.IO.Path]::GetExtension($zip) -ne ".zip") {
    throw "ZipPath must have a .zip extension: $zip"
}
if (-not (Test-Path -LiteralPath $zip -PathType Leaf)) {
    throw "ZipPath must be a file: $zip"
}

if (-not (Test-Path -LiteralPath (Join-Path $data "test.csv") -PathType Leaf) -or
    -not (Test-Path -LiteralPath (Join-Path $data "sample_submission.csv") -PathType Leaf)) {
    throw "DataDir must contain test.csv and sample_submission.csv: $data"
}

$rootWithSeparator = $projectRoot.TrimEnd('\') + '\'
if (-not $zip.StartsWith($rootWithSeparator, [StringComparison]::OrdinalIgnoreCase)) {
    throw "ZipPath must be inside the project root so Docker can build it: $projectRoot"
}
$relativeZip = $zip.Substring($rootWithSeparator.Length).Replace('\', '/')

docker version | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Docker engine is unavailable. Start Docker Desktop and wait until it reports Engine running."
}

if (-not $NoBuild) {
    docker build --progress=plain `
        --build-arg "SUBMISSION_ZIP=$relativeZip" `
        --file (Join-Path $PSScriptRoot "Dockerfile") `
        --tag $ImageName `
        $projectRoot
    if ($LASTEXITCODE -ne 0) { throw "Docker image build failed." }
}

$dockerArgs = @(
    "run", "--rm", "--network", "none",
    "--cpus", "6", "--cpuset-cpus", "0-5",
    "--memory", "28g", "--shm-size", "4g",
    "--pids-limit", "1024",
    "--mount", "type=bind,src=$zip,dst=/work/submission.zip,readonly",
    "--mount", "type=bind,src=$data,dst=/work/data-source,readonly"
)
if ($Gpu) { $dockerArgs += @("--gpus", "all") }
$dockerArgs += @($ImageName, "/work/submission.zip", "/work/data-source", "$TimeoutSeconds")

& docker @dockerArgs
if ($LASTEXITCODE -ne 0) { throw "Submission validation failed (exit code $LASTEXITCODE)." }
