$root = Split-Path -Parent -Path $MyInvocation.MyCommand.Path
$translationsDir = Join-Path -Path $root -ChildPath "..\translations"
$sourceDir = Join-Path -Path $root -ChildPath "..\src"
$uiFiles = Get-ChildItem -Path $sourceDir -Filter "*.ui" -File | Select-Object -ExpandProperty FullName
$mainWindowFile = Join-Path -Path $sourceDir -ChildPath "mainwindow.py"

# Get all .ts files in the translations directory
$tsFiles = Get-ChildItem -Path $translationsDir -Filter "*.ts" -File

# Include programmatic MainWindow strings as well as Qt Designer layouts.
foreach ($tsFile in $tsFiles) {
    pyside6-lupdate $mainWindowFile @uiFiles -ts $tsFile.FullName
}
