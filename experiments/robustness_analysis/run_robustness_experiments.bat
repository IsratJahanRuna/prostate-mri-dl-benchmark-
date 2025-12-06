@echo off
echo Starting Robustness Analysis Experiments
echo =======================================

:: Set paths (adjust these if needed)
set ROOT_DIR=%~dp0
set DATA_DIR=%ROOT_DIR%data
set OUTPUT_DIR=%ROOT_DIR%output_robustness

:: Create output directory if it doesn't exist
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo Data Directory: %DATA_DIR%
echo Output Directory: %OUTPUT_DIR%
echo.

:: ResNet Experiments
echo Running ResNet Experiments
echo ------------------------
echo Running ResNet with ADC...
python universal_trainer.py --model resnet --dataset ADC --data_root "%DATA_DIR%" --out_dir "%OUTPUT_DIR%"
if errorlevel 1 goto error
echo.

echo Running ResNet with DWI...
python universal_trainer.py --model resnet --dataset DWI --data_root "%DATA_DIR%" --out_dir "%OUTPUT_DIR%"
if errorlevel 1 goto error
echo.

echo Running ResNet with T2...
python universal_trainer.py --model resnet --dataset T2 --data_root "%DATA_DIR%" --out_dir "%OUTPUT_DIR%"
if errorlevel 1 goto error
echo.

:: DenseNet Experiments
echo Running DenseNet Experiments
echo --------------------------
echo Running DenseNet with ADC...
python universal_trainer.py --model densenet --dataset ADC --data_root "%DATA_DIR%" --out_dir "%OUTPUT_DIR%"
if errorlevel 1 goto error
echo.

echo Running DenseNet with DWI...
python universal_trainer.py --model densenet --dataset DWI --data_root "%DATA_DIR%" --out_dir "%OUTPUT_DIR%"
if errorlevel 1 goto error
echo.

echo Running DenseNet with T2...
python universal_trainer.py --model densenet --dataset T2 --data_root "%DATA_DIR%" --out_dir "%OUTPUT_DIR%"
if errorlevel 1 goto error
echo.

:: EfficientNet Experiments
echo Running EfficientNet Experiments
echo -----------------------------
echo Running EfficientNet with ADC...
python universal_trainer.py --model efficientnet --dataset ADC --data_root "%DATA_DIR%" --out_dir "%OUTPUT_DIR%"
if errorlevel 1 goto error
echo.

echo Running EfficientNet with DWI...
python universal_trainer.py --model efficientnet --dataset DWI --data_root "%DATA_DIR%" --out_dir "%OUTPUT_DIR%"
if errorlevel 1 goto error
echo.

echo Running EfficientNet with T2...
python universal_trainer.py --model efficientnet --dataset T2 --data_root "%DATA_DIR%" --out_dir "%OUTPUT_DIR%"
if errorlevel 1 goto error
echo.

echo All experiments completed successfully!
goto end

:error
echo.
echo Error occurred during execution!
echo Experiment failed! Check the error message above.
exit /b 1

:end
echo.
echo =======================================
echo All experiments completed!
echo Results are saved in: %OUTPUT_DIR%
pause