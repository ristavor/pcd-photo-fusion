from src.calibrator import read_cam_to_cam, read_velo_to_cam
from pathlib import Path
from src.loader import load_image, load_velodyne
import cv2
from src.colorize import project_points
import open3d as o3d
from src.colorize import assign_colors
# в начале main.py
ROOT = Path(__file__).resolve().parent.parent
calib_cam = ROOT / 'data/2011_09_28_calib/2011_09_28/calib_cam_to_cam.txt'
cam = read_cam_to_cam(calib_cam)
K = cam['P_rect_02'].reshape(3,4)[:3,:3]
calib_velo = ROOT / 'data/2011_09_28_calib/2011_09_28/calib_velo_to_cam.txt'
R, T = read_velo_to_cam(calib_velo)
print("K =", K)
print("R =", R)
print("T =", T)
img_path   = ROOT / 'data/2011_09_28_drive_0034_sync/image_02/data/0000000000.png'
velo_path  = ROOT / 'data/2011_09_28_drive_0034_sync/velodyne_points/data/0000000000.bin'

# Загрузка
img = load_image(img_path)
pts = load_velodyne(velo_path)

# Вывод для проверки
print(f"Изображение загружено: shape = {img.shape}, dtype = {img.dtype}")
print(f"В облаке точек: {pts.shape[0]} точек, первые 5 точек:\n{pts[:5]}")
# Берём только XYZ из pts (отбрасываем intensity)
xyz = pts[:, :3]

# Проецируем
uvz = project_points(xyz, R, T, K)

# Фильтруем только видимые точки
h, w = img.shape[:2]
mask = (uvz[:,2]>0) & (uvz[:,0]>=0) & (uvz[:,0]<w) & (uvz[:,1]>=0) & (uvz[:,1]<h)
uv = uvz[mask][:, :2].astype(int)

# Нарисуем первые 100 точек на копии изображения
# ------------------------------------------------------------------------------
# Заменяем отрисовку 100 точек на полупрозрачный оверлей всех видимых точек
# ------------------------------------------------------------------------------
# Создаём оверлей поверх оригинального кадра
# Показываем оригинальное изображение (без точек)
cv2.namedWindow('Projection Test', cv2.WINDOW_NORMAL)
cv2.imshow('Projection Test', img)
# НЕ вызываем waitKey(0) здесь, чтобы не блокировать дальнейший код

colors = assign_colors(uv, img)  # (M,3)

# 2) Создаём цветное облако точек
pcd_colored = o3d.geometry.PointCloud()
# Точки, отфильтрованные по mask, это xyz[mask]
pcd_colored.points = o3d.utility.Vector3dVector(xyz[mask])
pcd_colored.colors = o3d.utility.Vector3dVector(colors)

# 3) Визуализируем результат
o3d.visualization.draw_geometries(
    [pcd_colored],
    window_name='Colored Point Cloud',
    width=800,
    height=600
)
print("Закройте окно 'Colored Point Cloud', затем нажмите Enter в консоли для выхода.")
input()
cv2.destroyAllWindows()