@echo off
echo Starting execution of all ML model training commands...
echo.

echo [1/9] Running DenseNet on ADC dataset...
python densenet_universal.py    --dataset ADC --data_root "data" --out_dir "outputs/densenet/ADC"     --epochs 20 --seed 42 --batch_size 32 --lr 1e-3
if %errorlevel% neq 0 (
    echo Error in DenseNet ADC training
    pause
    exit /b 1
)

echo.
echo [2/9] Running DenseNet on DWI dataset...
python densenet_universal.py    --dataset DWI --data_root "data" --out_dir "outputs/densenet/DWI"     --epochs 20 --seed 42 --batch_size 32 --lr 1e-3
if %errorlevel% neq 0 (
    echo Error in DenseNet DWI training
    pause
    exit /b 1
)

echo.
echo [3/9] Running DenseNet on T2 dataset...
python densenet_universal.py    --dataset T2  --data_root "data" --out_dir "outputs/densenet/T2"     --epochs 20 --seed 42 --batch_size 32 --lr 1e-3
if %errorlevel% neq 0 (
    echo Error in DenseNet T2 training
    pause
    exit /b 1
)

echo.
echo [4/9] Running ResNet on ADC dataset...
python resnet_universal.py      --dataset ADC --data_root "data" --out_dir "outputs/resnet/ADC"       --epochs 20 --seed 42 --batch_size 32 --lr 1e-3
if %errorlevel% neq 0 (
    echo Error in ResNet ADC training
    pause
    exit /b 1
)

echo.
echo [5/9] Running ResNet on DWI dataset...
python resnet_universal.py      --dataset DWI --data_root "data" --out_dir "outputs/resnet/DWI"       --epochs 20 --seed 42 --batch_size 32 --lr 1e-3
if %errorlevel% neq 0 (
    echo Error in ResNet DWI training
    pause
    exit /b 1
)

echo.
echo [6/9] Running ResNet on T2 dataset...
python resnet_universal.py      --dataset T2  --data_root "data" --out_dir "outputs/resnet/T2"       --epochs 20 --seed 42 --batch_size 32 --lr 1e-3
if %errorlevel% neq 0 (
    echo Error in ResNet T2 training
    pause
    exit /b 1
)

echo.
echo [7/9] Running EfficientNet on ADC dataset...
python efficientnet_universal.py --dataset ADC --data_root "data" --out_dir "outputs/efficientnet/ADC" --epochs 20 --seed 42 --batch_size 32 --lr 1e-3
if %errorlevel% neq 0 (
    echo Error in EfficientNet ADC training
    pause
    exit /b 1
)

echo.
echo [8/9] Running EfficientNet on DWI dataset...
python efficientnet_universal.py --dataset DWI --data_root "data" --out_dir "outputs/efficientnet/DWI" --epochs 20 --seed 42 --batch_size 32 --lr 1e-3
if %errorlevel% neq 0 (
    echo Error in EfficientNet DWI training
    pause
    exit /b 1
)

echo.
echo [9/9] Running EfficientNet on T2 dataset...
python efficientnet_universal.py --dataset T2  --data_root "data" --out_dir "outputs/efficientnet/T2" --epochs 20 --seed 42 --batch_size 32 --lr 1e-3
if %errorlevel% neq 0 (
    echo Error in EfficientNet T2 training
    pause
    exit /b 1
)

echo.
echo All commands completed successfully!
pause
