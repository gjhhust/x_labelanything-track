@echo off
setlocal enabledelayedexpansion

echo __appname__ = "X-AnyLabeling" > anylabeling/app_info.py
echo __appdescription__ = "Advanced Auto Labeling Solution with Added Features" >> anylabeling/app_info.py
echo __version__ = "2.1.0" >> anylabeling/app_info.py
echo __preferred_device__ = "CPU"  ^# GPU or CPU >> anylabeling/app_info.py

set "system=%1"
if "%system%"=="win-cpu" (
    set "expected_value=CPU"
    set "variable_value=CPU"
    if "%variable_value%"=="%expected_value%" (
        pyinstaller --noconfirm anylabeling-win-cpu.spec
    ) else (
        echo Variable __preferred_device__ has a different value: %variable_value% (expected: %expected_value%)
    )
) else (
    echo System value '%system%' is not recognized.
)
