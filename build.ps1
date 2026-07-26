param(
    [string]$EntryFile = "main.tex",
    [string]$OutDir = "build",
    [int]$Resolution = 200,
    [switch]$OpenPdf
)

$ErrorActionPreference = "Stop"

function Resolve-Command {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [string[]]$FallbackRoots = @(),
        [string]$InstallHint
    )

    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }

    foreach ($root in $FallbackRoots) {
        if (-not $root -or -not (Test-Path $root)) {
            continue
        }

        $exe = Get-ChildItem -Path $root -Recurse -Filter $Name -ErrorAction SilentlyContinue |
            Select-Object -First 1

        if ($exe) {
            return $exe.FullName
        }
    }

    $message = "Required command '$Name' was not found."
    if ($InstallHint) {
        $message += " $InstallHint"
    }
    throw $message
}

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$entryPath = Join-Path $repoRoot $EntryFile

if (-not (Test-Path $entryPath)) {
    throw "Entry file '$EntryFile' was not found in '$repoRoot'."
}

$tectonicExe = Resolve-Command `
    -Name "tectonic.exe" `
    -FallbackRoots @((Join-Path $env:LOCALAPPDATA "Programs\Tectonic")) `
    -InstallHint "Install Tectonic first."

$pdftoppmExe = Resolve-Command `
    -Name "pdftoppm.exe" `
    -FallbackRoots @((Join-Path $env:LOCALAPPDATA "Programs\Poppler")) `
    -InstallHint "Install Poppler first."

$resolvedOutDir = Join-Path $repoRoot $OutDir
$pagesDir = Join-Path $resolvedOutDir "pages"
$entryStem = [System.IO.Path]::GetFileNameWithoutExtension($entryPath)
$pdfPath = Join-Path $resolvedOutDir ($entryStem + ".pdf")
$pagePrefix = Join-Path $pagesDir $entryStem

New-Item -ItemType Directory -Force -Path $resolvedOutDir | Out-Null
New-Item -ItemType Directory -Force -Path $pagesDir | Out-Null

Get-ChildItem -Path $pagesDir -Filter ($entryStem + "-*.png") -ErrorAction SilentlyContinue |
    Remove-Item -Force -ErrorAction SilentlyContinue

Push-Location $repoRoot
try {
    $tectonicArgs = @(
        "--keep-logs",
        "--keep-intermediates",
        "--outdir", $resolvedOutDir,
        $entryPath
    )

    Write-Host "Compiling PDF with Tectonic..."
    & $tectonicExe @tectonicArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Tectonic failed with exit code $LASTEXITCODE."
    }

    if (-not (Test-Path $pdfPath)) {
        throw "Expected PDF was not created: $pdfPath"
    }

    $pdftoppmArgs = @(
        "-png",
        "-r", $Resolution,
        $pdfPath,
        $pagePrefix
    )

    Write-Host "Exporting PDF pages to PNG..."
    & $pdftoppmExe @pdftoppmArgs
    if ($LASTEXITCODE -ne 0) {
        throw "pdftoppm failed with exit code $LASTEXITCODE."
    }

    $renderedPages = Get-ChildItem -Path $pagesDir -Filter ($entryStem + "-*.png") | Sort-Object Name

    Write-Host ""
    Write-Host "Build complete."
    Write-Host "Tectonic: $tectonicExe"
    Write-Host "pdftoppm: $pdftoppmExe"
    Write-Host "PDF:   $pdfPath"
    Write-Host "Pages: $pagesDir"
    Write-Host ("Count: " + $renderedPages.Count)

    if ($OpenPdf) {
        Start-Process $pdfPath | Out-Null
    }
}
finally {
    Pop-Location
}
