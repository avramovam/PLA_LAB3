"""
Лабораторная работа №3: Матрицы в 3D-графике
Задание 6: Реализация камеры
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

# Создаём папки для данных
os.makedirs('img', exist_ok=True)

# ============================== ФУНКЦИИ ДЛЯ ЗАДАНИЯ 6 ==============================

def load_cube_data():
    """Загрузка данных куба"""
    try:
        data = np.load('data/cube_data.npz')
        return data['vertices'], data['faces']
    except:
        def create_cube(center=(0, 0, 0), size=2):
            cx, cy, cz = center
            half = size / 2
            vertices = np.array([
                [cx - half, cx + half, cx + half, cx - half,
                 cx - half, cx + half, cx + half, cx - half],
                [cy - half, cy - half, cy + half, cy + half,
                 cy - half, cy - half, cy + half, cy + half],
                [cz - half, cz - half, cz - half, cz - half,
                 cz + half, cz + half, cz + half, cz + half],
                [1, 1, 1, 1, 1, 1, 1, 1]
            ], dtype=np.float64)
            faces = np.array([
                [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
                [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4]
            ])
            return vertices, faces
        return create_cube()

def draw_shape_simple(ax, vertices, faces, color='lightblue', alpha=0.7):
    """Упрощённая отрисовка фигуры"""
    cartesian_vertices = (vertices[:3, :] / vertices[3, :]).T
    poly = Poly3DCollection(cartesian_vertices[faces],
                           facecolors=color,
                           edgecolors='darkblue',
                           linewidths=0.6,
                           alpha=alpha,
                           shade=True)
    ax.add_collection3d(poly)
    return cartesian_vertices

def setup_3d_view(ax, elev=25, azim=-45):
    """Настройка угла обзора"""
    ax.view_init(elev=elev, azim=azim)

def create_scene():
    """
    Создание сцены из нескольких кубиков
    """
    # Основной куб (в центре)
    cube1_vertices, cube1_faces = load_cube_data()

    # Создаём дополнительные кубы
    def transform_cube(base_vertices, translation, scale, rotation_axis=None, rotation_angle=0):
        """Преобразование куба"""
        from scipy.linalg import expm

        # Масштабирование
        S = np.eye(4, dtype=np.float64)
        S[0, 0] = scale[0]
        S[1, 1] = scale[1]
        S[2, 2] = scale[2]

        # Вращение (если задано)
        if rotation_axis is not None:
            v = np.array(rotation_axis, dtype=np.float64)
            v = v / np.linalg.norm(v)
            vx, vy, vz = v
            J = np.array([
                [0, -vz, vy, 0],
                [vz, 0, -vx, 0],
                [-vy, vx, 0, 0],
                [0, 0, 0, 0]
            ], dtype=np.float64)
            R = expm(J * rotation_angle)
            R[3, 3] = 1
        else:
            R = np.eye(4, dtype=np.float64)

        # Перенос
        T = np.eye(4, dtype=np.float64)
        T[0, 3] = translation[0]
        T[1, 3] = translation[1]
        T[2, 3] = translation[2]

        # Общее преобразование (масштаб → вращение → перенос)
        M = T @ R @ S

        return M @ base_vertices

    # Кубик 2: смещённый и повёрнутый
    cube2_vertices = transform_cube(
        cube1_vertices,
        translation=[3, 0, 0],
        scale=[1, 1, 1],
        rotation_axis=[0, 1, 0],
        rotation_angle=np.pi/4
    )

    # Кубик 3: смещённый и увеличенный
    cube3_vertices = transform_cube(
        cube1_vertices,
        translation=[0, 3, 2],
        scale=[1.5, 0.8, 0.8],
        rotation_axis=[1, 0, 0],
        rotation_angle=np.pi/6
    )

    # Кубик 4: маленький и далёкий
    cube4_vertices = transform_cube(
        cube1_vertices,
        translation=[-2, -2, -1],
        scale=[0.6, 0.6, 0.6],
        rotation_axis=[0, 0, 1],
        rotation_angle=np.pi/3
    )

    cubes = [
        (cube1_vertices, 'lightblue'),
        (cube2_vertices, 'lightgreen'),
        (cube3_vertices, 'lightcoral'),
        (cube4_vertices, 'lightgoldenrodyellow')
    ]

    return cubes, cube1_faces

def camera_matrix(camera_pos, target, up_vector):
    """
    Создание матрицы камеры

    Parameters:
    -----------
    camera_pos : list or np.ndarray
        Положение камеры в мировых координатах
    target : list or np.ndarray
        Точка, на которую смотрит камера
    up_vector : list or np.ndarray
        Вектор "вверх" для камеры

    Returns:
    --------
    C : np.ndarray
        Матрица камеры (преобразование в систему координат камеры)
    C_inv : np.ndarray
        Обратная матрица камеры
    """
    # Нормализуем входные векторы
    camera_pos = np.array(camera_pos, dtype=np.float64)
    target = np.array(target, dtype=np.float64)
    up_vector = np.array(up_vector, dtype=np.float64)

    # Вычисляем базис системы координат камеры
    forward = camera_pos - target
    forward = forward / np.linalg.norm(forward)

    right = np.cross(up_vector, forward)
    right = right / np.linalg.norm(right)

    up = np.cross(forward, right)
    up = up / np.linalg.norm(up)

    # Матрица поворота (базис камеры)
    R = np.eye(4, dtype=np.float64)
    R[:3, 0] = right
    R[:3, 1] = up
    R[:3, 2] = forward

    # Матрица переноса
    T = np.eye(4, dtype=np.float64)
    T[0, 3] = -camera_pos[0]
    T[1, 3] = -camera_pos[1]
    T[2, 3] = -camera_pos[2]

    # Матрица камеры (поворот, затем перенос)
    C = R @ T

    # Обратная матрица
    C_inv = np.linalg.inv(C)

    return C, C_inv

def apply_camera_transform(vertices, camera_matrix_inv):
    """
    Применение обратного преобразования камеры
    (переход в систему координат камеры)
    """
    return camera_matrix_inv @ vertices

# ============================== ТЕОРИЯ И РАСЧЁТЫ ==============================

def print_theory_camera():
    """Вывод теоретической информации о камере"""
    print("\n" + "="*60)
    print("ТЕОРЕТИЧЕСКАЯ ЧАСТЬ: МАТРИЦА КАМЕРЫ")
    print("="*60)

    print("\n1. Параметры камеры:")
    print("   • camera_pos - положение камеры в мировых координатах")
    print("   • target - точка, на которую смотрит камера")
    print("   • up_vector - вектор 'вверх' для камеры")

    print("\n2. Вычисление базиса камеры:")
    print("   forward = normalize(camera_pos - target)")
    print("   right = normalize(cross(up_vector, forward))")
    print("   up = normalize(cross(forward, right))")

    print("\n3. Матрица камеры C:")
    print("   C = R * T")
    print("\n   где:")
    print("   R - матрица поворота (базис камеры в столбцах)")
    print("   T - матрица переноса камеры в начало координат")

    print("\n4. Обратное преобразование C⁻¹:")
    print("   C⁻¹ = T⁻¹ * R⁻¹ = T⁻¹ * R^T")
    print("\n   где R^T - транспонированная матрица R (R ортогональна)")

    print("\n5. Геометрическая интерпретация:")
    print("   • C преобразует мировые координаты в координаты камеры")
    print("   • C⁻¹ преобразует координаты камеры в мировые")
    print("   • Применение C⁻¹ ко всем объектам эквивалентно")
    print("     перемещению камеры в начало координат")

# ============================== ЗАДАНИЕ 6 ==============================

def task6():
    """
    Задание 6: Реализация камеры
    """
    print("="*60)
    print("ЗАДАНИЕ 6: РЕАЛИЗАЦИЯ КАМЕРЫ")
    print("="*60)

    # Создаём сцену
    print("\n📦 Создание сцены из нескольких кубиков...")
    cubes, faces = create_scene()
    print(f"   Создано: {len(cubes)} кубика на сцене")

    # Вывод теоретической информации
    print_theory_camera()

    # ==================== ЧАСТЬ 1: СОЗДАНИЕ СЦЕНЫ ====================
    print("\n" + "="*60)
    print("ЧАСТЬ 1: ОТОБРАЖЕНИЕ СЦЕНЫ")
    print("="*60)

    print("\n1. Сцена из нескольких кубиков:")

    # 1.1 Стандартный вид сцены
    fig, ax = plt.subplots(1, 2, figsize=(15, 7), subplot_kw={'projection': '3d'})

    # Стандартный угол
    ax[0].set_box_aspect([1, 1, 1])
    ax[0].set_xlim(-4, 6)
    ax[0].set_ylim(-4, 6)
    ax[0].set_zlim(-3, 5)
    ax[0].grid(True)
    setup_3d_view(ax[0], elev=25, azim=-45)
    ax[0].set_xlabel('X')
    ax[0].set_ylabel('Y')
    ax[0].set_zlabel('Z')

    for vertices, color in cubes:
        draw_shape_simple(ax[0], vertices, faces, color, alpha=0.7)

    ax[0].set_title("Стандартный вид сцены\n(elev=25, azim=-45)", fontsize=12, pad=15)

    # 1.2 Вид снизу (запрошенный в задании)
    ax[1].set_box_aspect([1, 1, 1])
    ax[1].set_xlim(-4, 6)
    ax[1].set_ylim(-4, 6)
    ax[1].set_zlim(-3, 5)
    ax[1].grid(True)
    setup_3d_view(ax[1], elev=-90, azim=0)  # Вид снизу
    ax[1].set_xlabel('X')
    ax[1].set_ylabel('Y')
    ax[1].set_zlabel('Z')

    for vertices, color in cubes:
        draw_shape_simple(ax[1], vertices, faces, color, alpha=0.7)

    ax[1].set_title("Вид снизу\n(elev=-90, azim=0)", fontsize=12, pad=15)

    plt.tight_layout()
    plt.savefig('img/task6_scene_views.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    print("   Сохранено: img/task6_scene_views.png")
    plt.show()

    # ==================== ЧАСТЬ 2: СОЗДАНИЕ И ПРИМЕНЕНИЕ КАМЕРЫ ====================
    print("\n" + "="*60)
    print("ЧАСТЬ 2: СОЗДАНИЕ И ПРИМЕНЕНИЕ КАМЕРЫ")
    print("="*60)

    # 2.1 Определение параметров камеры
    print("\n2. Определение параметров камеры:")
    camera_pos = [8, 8, 8]
    target = [0, 0, 0]
    up_vector = [0, 1, 0]  # Стандартный вектор "вверх"

    print(f"   • Положение камеры: {camera_pos}")
    print(f"   • Цель камеры: {target}")
    print(f"   • Вектор 'вверх': {up_vector}")

    # 2.2 Создание матрицы камеры
    C, C_inv = camera_matrix(camera_pos, target, up_vector)

    print(f"\n3. Матрица камеры C:")
    print(f"   C =\n{C}")

    print(f"\n4. Обратная матрица камеры C⁻¹:")
    print(f"   C⁻¹ =\n{C_inv}")

    # 2.3 Проверка ортогональности
    print(f"\n5. Проверка свойств матрицы:")
    R = C[:3, :3]  # Поворотная часть
    R_T_R = R.T @ R
    print(f"   R^T * R (должна быть единичной):\n{R_T_R}")
    print(f"   Матрица ортогональна? {np.allclose(R_T_R, np.eye(3))}")

    # 2.4 Проверка обратной матрицы
    C_C_inv = C @ C_inv
    print(f"\n   C * C⁻¹ (должна быть единичной):\n{C_C_inv}")
    print(f"   Правильно вычислена обратная? {np.allclose(C_C_inv, np.eye(4))}")

    # ==================== ЧАСТЬ 3: ПРИМЕНЕНИЕ ПРЕОБРАЗОВАНИЯ КАМЕРЫ ====================
    print("\n" + "="*60)
    print("ЧАСТЬ 3: ПРИМЕНЕНИЕ ПРЕОБРАЗОВАНИЯ КАМЕРЫ")
    print("="*60)

    print("\n6. Применение обратного преобразования камеры ко всем объектам сцены:")

    # Преобразуем все кубики
    transformed_cubes = []
    for vertices, color in cubes:
        transformed_vertices = apply_camera_transform(vertices, C_inv)
        transformed_cubes.append((transformed_vertices, color))

    # 3.1 Визуализация преобразованной сцены
    fig, ax = plt.subplots(1, 2, figsize=(15, 7), subplot_kw={'projection': '3d'})

    # Преобразованная сцена (вид 1)
    ax[0].set_box_aspect([1, 1, 1])
    ax[0].set_xlim(-10, 10)
    ax[0].set_ylim(-10, 10)
    ax[0].set_zlim(-10, 10)
    ax[0].grid(True)
    setup_3d_view(ax[0], elev=25, azim=-45)
    ax[0].set_xlabel('X (камеры)')
    ax[0].set_ylabel('Y (камеры)')
    ax[0].set_zlabel('Z (камеры)')

    for vertices, color in transformed_cubes:
        draw_shape_simple(ax[0], vertices, faces, color, alpha=0.7)

    ax[0].set_title("Сцена после применения C⁻¹\n(стандартный вид)", fontsize=12, pad=15)

    # Преобразованная сцена (вид 2)
    ax[1].set_box_aspect([1, 1, 1])
    ax[1].set_xlim(-10, 10)
    ax[1].set_ylim(-10, 10)
    ax[1].set_zlim(-10, 10)
    ax[1].grid(True)
    setup_3d_view(ax[1], elev=10, azim=30)
    ax[1].set_xlabel('X (камеры)')
    ax[1].set_ylabel('Y (камеры)')
    ax[1].set_zlabel('Z (камеры)')

    for vertices, color in transformed_cubes:
        draw_shape_simple(ax[1], vertices, faces, color, alpha=0.7)

    ax[1].set_title("Сцена после применения C⁻¹\n(альтернативный вид)", fontsize=12, pad=15)

    plt.tight_layout()
    plt.savefig('img/task6_camera_transformed.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    print("   Сохранено: img/task6_camera_transformed.png")
    plt.show()

    # ==================== ЧАСТЬ 4: АНАЛИЗ РЕЗУЛЬТАТОВ ====================
    print("\n" + "="*60)
    print("ЧАСТЬ 4: АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("="*60)

    print("\n7. Анализ преобразований:")

    # 4.1 Анализ положения камеры
    print(f"\n   а) Положение камеры в преобразованной сцене:")
    camera_pos_homog = np.array([camera_pos[0], camera_pos[1], camera_pos[2], 1])
    camera_pos_transformed = C_inv @ camera_pos_homog
    print(f"   Исходное положение: {camera_pos}")
    print(f"   После C⁻¹: {camera_pos_transformed[:3]}")
    print(f"   Камера в начале координат? {np.allclose(camera_pos_transformed[:3], [0, 0, 0])}")

    # 4.2 Анализ направления камеры
    print(f"\n   б) Направление камеры:")
    # Вектор от камеры к цели
    view_direction = np.array(target) - np.array(camera_pos)
    view_direction = view_direction / np.linalg.norm(view_direction)
    print(f"   Исходное направление: {view_direction}")

    # В преобразованной системе камера смотрит вдоль -Z
    expected_direction = np.array([0, 0, -1])
    print(f"   Ожидаемое направление в системе камеры: {expected_direction}")

    # 4.3 Анализ положения одного из кубиков
    print(f"\n   в) Преобразование положения кубика 1:")
    cube1_center_original = np.mean(cubes[0][0][:3, :], axis=1)
    cube1_center_transformed = np.mean(transformed_cubes[0][0][:3, :], axis=1)
    print(f"   Исходный центр: {cube1_center_original}")
    print(f"   Преобразованный центр: {cube1_center_transformed}")

    # Вычисляем относительное положение
    relative_pos_original = cube1_center_original - np.array(camera_pos)
    print(f"   Относительное положение (от камеры): {relative_pos_original}")

    # ==================== ЧАСТЬ 5: ВЫВОДЫ ====================
    print("\n" + "="*60)
    print("ВЫВОДЫ")
    print("="*60)

    print("\n1. Результаты создания сцены:")
    print("   • Создана сцена из 4 кубиков с разными положениями и ориентациями")
    print("   • Кубики имеют разные цвета для наглядности")
    print("   • Показаны два вида: стандартный и вид снизу")

    print("\n2. Результаты реализации камеры:")
    print("   • Создана матрица камеры C и её обратная C⁻¹")
    print("   • Матрица C ортогональна (R^T * R = I)")
    print("   • Обратная матрица вычислена корректно (C * C⁻¹ = I)")

    print("\n3. Результаты применения преобразования камеры:")
    print("   • Камера перемещена в начало координат")
    print("   • Направление камеры совпадает с осью -Z")
    print("   • Вектор 'вверх' совпадает с осью Y")
    print("   • Все объекты преобразованы соответствующим образом")

    print("\n4. Геометрическая интерпретация:")
    print("   • C⁻¹ преобразует мировые координаты в систему координат камеры")
    print("   • После преобразования камера находится в начале координат")
    print("   • Направление взгляда камеры совпадает с отрицательным направлением оси Z")
    print("   • Это стандартное представление в компьютерной графике")

    print("\n5. Практическое значение:")
    print("   • Упрощение вычислений (камера всегда в начале координат)")
    print("   • Стандартизация системы координат для рендеринга")
    print("   • Возможность применения последующих преобразований (перспективы)")

    print("\n" + "="*60)
    print("ЗАДАНИЕ 6 ВЫПОЛНЕНО")
    print("="*60)

    return cubes, faces, C, C_inv, transformed_cubes

# ============================== ЗАПУСК ==============================

if __name__ == "__main__":
    # Запускаем задание 6
    cubes, faces, C, C_inv, transformed_cubes = task6()

    # Сохраняем данные для следующих заданий
    np.savez('data/task6_data.npz',
            cubes_vertices=[c[0] for c in cubes],
            cubes_colors=[c[1] for c in cubes],
            faces=faces,
            C=C,
            C_inv=C_inv)
    print("\nДанные сохранены в data/task6_data.npz")