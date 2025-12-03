"""
Лабораторная работа №3: Матрицы в 3D-графике
Задание 4: Вращение кубика вокруг произвольной оси
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.linalg import expm
import os

# Создаём папки для данных
os.makedirs('img', exist_ok=True)


# ============================== ФУНКЦИИ ДЛЯ ЗАДАНИЯ 4 ==============================

def load_previous_data():
    """Загрузка данных из предыдущих заданий"""
    try:
        data = np.load('data/cube_data.npz')
        return data['vertices'], data['faces']
    except:
        # Если файла нет, создаём куб
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


def draw_shape_with_axis(ax, vertices, faces, color='lightblue', alpha=0.7,
                         axis_vector=None, axis_color='red', axis_length=2):
    """Отрисовка фигуры с осью вращения"""
    cartesian_vertices = (vertices[:3, :] / vertices[3, :]).T
    poly = Poly3DCollection(cartesian_vertices[faces],
                            facecolors=color,
                            edgecolors='darkblue',
                            linewidths=0.8,
                            alpha=alpha,
                            shade=True)
    ax.add_collection3d(poly)

    # Если задана ось, рисуем её
    if axis_vector is not None:
        axis_vector = np.array(axis_vector)
        axis_vector = axis_vector / np.linalg.norm(axis_vector) * axis_length
        ax.quiver(0, 0, 0,
                  axis_vector[0], axis_vector[1], axis_vector[2],
                  color=axis_color, linewidth=2, arrow_length_ratio=0.1)
        ax.text(axis_vector[0] * 1.1, axis_vector[1] * 1.1, axis_vector[2] * 1.1,
                f'v', color=axis_color, fontsize=10, fontweight='bold')

    return cartesian_vertices


def setup_3d_plot(figsize=(10, 8), elev=25, azim=-45,
                  limits=(-2, 2), grid=True):
    """
    Настройка 3D графика с координатной сеткой
    """
    fig = plt.figure(figsize=figsize, dpi=100, facecolor='white')
    ax = fig.add_subplot(111, projection='3d')

    # Настраиваем пропорции осей
    ax.set_box_aspect([1, 1, 1])

    # Настраиваем пределы
    if isinstance(limits[0], (tuple, list)):
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])
        ax.set_zlim(limits[2])
    else:
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_zlim(limits)

    # Включаем координатную сетку
    ax.grid(grid)

    # Устанавливаем угол обзора
    ax.view_init(elev=elev, azim=azim)

    # Оставляем подписи осей для ориентира
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    return fig, ax


def rotation_matrix_x(theta):
    """
    Матрица вращения вокруг оси X
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1]
    ], dtype=np.float64)


def rotation_matrix_y(theta):
    """
    Матрица вращения вокруг оси Y
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-s, 0, c, 0],
        [0, 0, 0, 1]
    ], dtype=np.float64)


def rotation_matrix_z(theta):
    """
    Матрица вращения вокруг оси Z
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0, 0],
        [s, c, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float64)


def rotation_matrix_axis(v, theta):
    """
    Матрица вращения вокруг произвольной оси через начало координат
    Использует формулу через матричную экспоненту

    Parameters:
    -----------
    v : np.ndarray или list
        Вектор оси вращения (не обязательно единичный)
    theta : float
        Угол вращения в радианах

    Returns:
    --------
    R : np.ndarray
        Матрица вращения 4x4
    """
    # Нормализуем вектор оси
    v = np.array(v, dtype=np.float64)
    v = v / np.linalg.norm(v)

    # Создаём кососимметрическую матрицу
    vx, vy, vz = v
    J = np.array([
        [0, -vz, vy, 0],
        [vz, 0, -vx, 0],
        [-vy, vx, 0, 0],
        [0, 0, 0, 0]
    ], dtype=np.float64)

    # Вычисляем матричную экспоненту
    R = expm(J * theta)
    R[3, 3] = 1  # Восстанавливаем последний элемент

    return R


def apply_transformation(vertices, transformation_matrix):
    """
    Применение матрицы преобразования к вершинам
    """
    return transformation_matrix @ vertices


# ============================== ТЕОРИЯ И РАСЧЁТЫ ==============================

def print_theory_rotation():
    """Вывод теоретической информации о вращении"""
    print("\n" + "=" * 60)
    print("ТЕОРЕТИЧЕСКАЯ ЧАСТЬ: МАТРИЦЫ ВРАЩЕНИЯ")
    print("=" * 60)

    print("\n1. Матрицы вращения вокруг осей координат:")

    print("\n   а) Вращение вокруг оси X на угол θ:")
    print("      ⎡ 1     0      0    0 ⎤")
    print("      ⎢ 0   cosθ   -sinθ  0 ⎥")
    print("      ⎢ 0   sinθ    cosθ  0 ⎥")
    print("      ⎣ 0     0      0    1 ⎦")

    print("\n   б) Вращение вокруг оси Y на угол θ:")
    print("      ⎡ cosθ   0    sinθ   0 ⎤")
    print("      ⎢   0    1      0    0 ⎥")
    print("      ⎢-sinθ   0    cosθ   0 ⎥")
    print("      ⎣   0    0      0    1 ⎦")

    print("\n   в) Вращение вокруг оси Z на угол θ:")
    print("      ⎡ cosθ   -sinθ   0   0 ⎤")
    print("      ⎢ sinθ    cosθ   0   0 ⎥")
    print("      ⎢   0       0    1   0 ⎥")
    print("      ⎣   0       0    0   1 ⎦")

    print("\n2. Вращение вокруг произвольной оси v:")
    print("   Пусть v = (v_x, v_y, v_z) - единичный вектор оси")
    print("   Строим кососимметрическую матрицу J:")
    print("        ⎡  0   -v_z   v_y  0 ⎤")
    print("   J =  ⎢ v_z    0   -v_x  0 ⎥")
    print("        ⎢-v_y   v_x    0   0 ⎥")
    print("        ⎣  0     0     0   0 ⎦")
    print("   Тогда матрица вращения: R_v(θ) = e^{Jθ}")

    print("\n3. Свойства матриц вращения:")
    print("   • Ортогональность: R^T * R = I")
    print("   • Определитель: det(R) = 1")
    print("   • Собственное значение 1 (ось вращения)")
    print("   • Не коммутируют в общем случае: R1 * R2 ≠ R2 * R1")

    print("\n4. Формула Родрига (альтернатива матричной экспоненте):")
    print("   R_v(θ) = I + sinθ * J + (1 - cosθ) * J^2")
    print("   где J - кососимметрическая матрица, построенная из v")


# ============================== ЗАДАНИЕ 4 ==============================

def task4():
    """
    Задание 4: Вращение кубика вокруг произвольной оси
    """
    print("=" * 60)
    print("ЗАДАНИЕ 4: ВРАЩЕНИЕ КУБИКА ВОКРУГ ПРОИЗВОЛЬНОЙ ОСИ")
    print("=" * 60)

    # Загружаем данные куба
    print("\n📦 Загрузка данных куба...")
    vertices, faces = load_previous_data()
    print(f"   Загружено: {vertices.shape[1]} вершин, {faces.shape[0]} граней")

    # Вывод теоретической информации
    print_theory_rotation()

    # ==================== ЧАСТЬ 1: ВРАЩЕНИЕ ВОКРУГ ОСЕЙ КООРДИНАТ ====================
    print("\n" + "=" * 60)
    print("ЧАСТЬ 1: ВРАЩЕНИЕ ВОКРУГ ОСЕЙ КООРДИНАТ")
    print("=" * 60)

    print("\n1. Вращение вокруг осей X, Y, Z на угол π/4 (45°):")

    # 1.1 Вращение вокруг оси X
    print("\n   а) Вращение вокруг оси X:")
    Rx = rotation_matrix_x(np.pi / 4)
    print(f"   Матрица R_x(π/4):\n{Rx}")

    vertices_Rx = apply_transformation(vertices, Rx)

    fig, ax = setup_3d_plot(figsize=(8, 6), elev=25, azim=-45, limits=(-1.5, 1.5), grid=True)
    draw_shape_with_axis(ax, vertices_Rx, faces, color='lightcoral', alpha=0.7,
                         axis_vector=[1, 0, 0], axis_color='red')
    ax.set_title("Вращение вокруг оси X на 45°", fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig('img/task4_rotation_x.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    print("   Сохранено: img/task4_rotation_x.png")
    plt.show()

    # 1.2 Вращение вокруг оси Y
    print("\n   б) Вращение вокруг оси Y:")
    Ry = rotation_matrix_y(np.pi / 4)
    print(f"   Матрица R_y(π/4):\n{Ry}")

    vertices_Ry = apply_transformation(vertices, Ry)

    fig, ax = setup_3d_plot(figsize=(8, 6), elev=25, azim=-45, limits=(-1.5, 1.5), grid=True)
    draw_shape_with_axis(ax, vertices_Ry, faces, color='lightgreen', alpha=0.7,
                         axis_vector=[0, 1, 0], axis_color='green')
    ax.set_title("Вращение вокруг оси Y на 45°", fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig('img/task4_rotation_y.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    print("   Сохранено: img/task4_rotation_y.png")
    plt.show()

    # 1.3 Вращение вокруг оси Z
    print("\n   в) Вращение вокруг оси Z:")
    Rz = rotation_matrix_z(np.pi / 4)
    print(f"   Матрица R_z(π/4):\n{Rz}")

    vertices_Rz = apply_transformation(vertices, Rz)

    fig, ax = setup_3d_plot(figsize=(8, 6), elev=25, azim=-45, limits=(-1.5, 1.5), grid=True)
    draw_shape_with_axis(ax, vertices_Rz, faces, color='lightblue', alpha=0.7,
                         axis_vector=[0, 0, 1], axis_color='blue')
    ax.set_title("Вращение вокруг оси Z на 45°", fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig('img/task4_rotation_z.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    print("   Сохранено: img/task4_rotation_z.png")
    plt.show()

    # ==================== ЧАСТЬ 2: ВРАЩЕНИЕ ВОКРУГ ПРОИЗВОЛЬНЫХ ОСЕЙ ====================
    print("\n" + "=" * 60)
    print("ЧАСТЬ 2: ВРАЩЕНИЕ ВОКРУГ ПРОИЗВОЛЬНЫХ ОСЕЙ")
    print("=" * 60)

    # 2.1 Первая произвольная ось
    print("\n2. Вращение вокруг произвольных осей:")
    print("\n   а) Первая произвольная ось: v1 = [1, 1, 0], угол θ1 = π/3 (60°)")
    v1 = [1, 1, 0]
    theta1 = np.pi / 3

    R1 = rotation_matrix_axis(v1, theta1)
    print(f"   Матрица R1 (вращение вокруг v1 на θ1):\n{R1}")

    vertices_R1 = apply_transformation(vertices, R1)

    fig, ax = setup_3d_plot(figsize=(8, 6), elev=25, azim=-45, limits=(-1.5, 1.5), grid=True)
    draw_shape_with_axis(ax, vertices_R1, faces, color='lightcoral', alpha=0.7,
                         axis_vector=v1, axis_color='red', axis_length=1.5)
    ax.set_title(f"Вращение вокруг оси v1={v1} на 60°", fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig('img/task4_rotation_v1.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    print("   Сохранено: img/task4_rotation_v1.png")
    plt.show()

    # 2.2 Вторая произвольная ось
    print("\n   б) Вторая произвольная ось: v2 = [0, 1, 1], угол θ2 = π/2 (90°)")
    v2 = [0, 1, 1]
    theta2 = np.pi / 2

    R2 = rotation_matrix_axis(v2, theta2)
    print(f"   Матрица R2 (вращение вокруг v2 на θ2):\n{R2}")

    vertices_R2 = apply_transformation(vertices, R2)

    fig, ax = setup_3d_plot(figsize=(8, 6), elev=25, azim=-45, limits=(-1.5, 1.5), grid=True)
    draw_shape_with_axis(ax, vertices_R2, faces, color='lightgreen', alpha=0.7,
                         axis_vector=v2, axis_color='green', axis_length=1.5)
    ax.set_title(f"Вращение вокруг оси v2={v2} на 90°", fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig('img/task4_rotation_v2.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    print("   Сохранено: img/task4_rotation_v2.png")
    plt.show()

    # ==================== ЧАСТЬ 3: КОМБИНИРОВАННЫЕ ВРАЩЕНИЯ ====================
    print("\n" + "=" * 60)
    print("ЧАСТЬ 3: КОМБИНИРОВАННЫЕ ВРАЩЕНИЯ")
    print("=" * 60)

    print("\n3. Комбинированные вращения:")

    # 3.1 Комбинация R1 * R2
    print("\n   а) Комбинация R12 = R1 * R2 (сначала R2, потом R1):")
    R12 = R1 @ R2
    print(f"   Матрица R12:\n{R12}")

    vertices_R12 = apply_transformation(vertices, R12)

    fig, ax = setup_3d_plot(figsize=(8, 6), elev=25, azim=-45, limits=(-1.5, 1.5), grid=True)
    draw_shape_with_axis(ax, vertices_R12, faces, color='lightgoldenrodyellow', alpha=0.7)
    ax.set_title("Комбинированное вращение: R12 = R1 * R2", fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig('img/task4_rotation_R12.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    print("   Сохранено: img/task4_rotation_R12.png")
    plt.show()

    # 3.2 Комбинация R2 * R1
    print("\n   б) Комбинация R21 = R2 * R1 (сначала R1, потом R2):")
    R21 = R2 @ R1
    print(f"   Матрица R21:\n{R21}")

    vertices_R21 = apply_transformation(vertices, R21)

    fig, ax = setup_3d_plot(figsize=(8, 6), elev=25, azim=-45, limits=(-1.5, 1.5), grid=True)
    draw_shape_with_axis(ax, vertices_R21, faces, color='lightpink', alpha=0.7)
    ax.set_title("Комбинированное вращение: R21 = R2 * R1", fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig('img/task4_rotation_R21.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    print("   Сохранено: img/task4_rotation_R21.png")
    plt.show()

    # 3.3 Проверка коммутативности
    print("\n   в) Проверка коммутативности вращений:")
    print(f"   R1 * R2 == R2 * R1? {np.allclose(R1 @ R2, R2 @ R1)}")
    print("   Матрицы не равны - вращения не коммутируют!")

    # 3.4 Сравнительная визуализация
    fig = plt.figure(figsize=(12, 5), dpi=100)

    # График R12
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_box_aspect([1, 1, 1])
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_zlim(-1.5, 1.5)
    ax1.grid(True)
    ax1.view_init(elev=25, azim=-45)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    draw_shape_with_axis(ax1, vertices_R12, faces, color='lightgoldenrodyellow', alpha=0.7)
    ax1.set_title("R12 = R1 * R2\n(сначала R2, потом R1)", fontsize=12, pad=15)

    # График R21
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_box_aspect([1, 1, 1])
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_zlim(-1.5, 1.5)
    ax2.grid(True)
    ax2.view_init(elev=25, azim=-45)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    draw_shape_with_axis(ax2, vertices_R21, faces, color='lightpink', alpha=0.7)
    ax2.set_title("R21 = R2 * R1\n(сначала R1, потом R2)", fontsize=12, pad=15)

    plt.tight_layout()
    plt.savefig('img/task4_rotation_comparison.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    print("\n   Сохранено: img/task4_rotation_comparison.png")
    plt.show()

    # ==================== ЧАСТЬ 4: АНАЛИЗ И ПРОВЕРКИ ====================
    print("\n" + "=" * 60)
    print("ЧАСТЬ 4: АНАЛИЗ И ПРОВЕРКИ")
    print("=" * 60)

    print("\n4. Проверка свойств матриц вращения:")

    # 4.1 Проверка ортогональности
    print("\n   а) Проверка ортогональности R1:")
    R1_3x3 = R1[:3, :3]
    I_check = R1_3x3.T @ R1_3x3
    print(f"   R1^T * R1 (должна быть единичной матрицей):\n{I_check}")
    print(f"   Матрица ортогональна? {np.allclose(I_check, np.eye(3))}")

    # 4.2 Проверка определителя
    print("\n   б) Проверка определителя R1:")
    det_R1 = np.linalg.det(R1_3x3)
    print(f"   det(R1) = {det_R1:.6f} (должно быть 1)")
    print(f"   Определитель равен 1? {np.isclose(det_R1, 1.0)}")

    # 4.3 Нахождение оси вращения
    print("\n   в) Нахождение оси вращения матрицы R1:")
    # Ось вращения - собственный вектор с собственным значением 1
    eigenvalues, eigenvectors = np.linalg.eig(R1_3x3)

    # Ищем собственное значение, близкое к 1
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    axis_from_matrix = np.real(eigenvectors[:, idx])

    # Нормализуем исходную ось для сравнения
    v1_normalized = np.array(v1) / np.linalg.norm(v1)

    print(f"   Исходная ось v1 (нормализованная): {v1_normalized}")
    print(f"   Ось из матрицы R1: {axis_from_matrix}")
    print(f"   Совпадение? {np.allclose(np.abs(v1_normalized), np.abs(axis_from_matrix))}")

    # ==================== ЧАСТЬ 5: ВЫВОДЫ ====================
    print("\n" + "=" * 60)
    print("ВЫВОДЫ")
    print("=" * 60)

    print("\n1. Свойства матриц вращения:")
    print("   • Матрицы вращения ортогональны: R^T * R = I")
    print("   • Определитель равен 1: det(R) = 1")
    print("   • Имеют собственное значение 1 (соответствует оси вращения)")
    print("   • Не коммутируют в общем случае: R1 * R2 ≠ R2 * R1")

    print("\n2. Вращение вокруг произвольной оси:")
    print("   • Ось задаётся единичным вектором v")
    print("   • Используется кососимметрическая матрица J(v)")
    print("   • Матрица вращения: R_v(θ) = e^{Jθ}")
    print("   • Можно также использовать формулу Родрига")

    print("\n3. Геометрическая интерпретация:")
    print("   • Вращение вокруг осей координат - частные случаи")
    print("   • Порядок вращений важен (некоммутативность)")
    print("   • Комбинированные вращения дают сложные ориентации")

    print("\n4. Практическое значение:")
    print("   • В компьютерной графике вращения часто комбинируются")
    print("   • Порядок важен: обычно yaw → pitch → roll")
    print("   • Матричная экспонента позволяет вращать вокруг произвольной оси")

    print("\n5. Результаты эксперимента:")
    print("   • R_x(45°): вращение вокруг оси X")
    print("   • R_y(45°): вращение вокруг оси Y")
    print("   • R_z(45°): вращение вокруг оси Z")
    print("   • R_v1(60°): вращение вокруг оси [1,1,0]")
    print("   • R_v2(90°): вращение вокруг оси [0,1,1]")
    print("   • R12 ≠ R21: подтверждение некоммутативности")

    print("\n" + "=" * 60)
    print("ЗАДАНИЕ 4 ВЫПОЛНЕНО")
    print("=" * 60)

    return vertices, faces, R1, R2, R12, R21


# ============================== ЗАПУСК ==============================

if __name__ == "__main__":
    # Запускаем задание 4
    vertices, faces, R1, R2, R12, R21 = task4()

    # Сохраняем данные для следующих заданий
    np.savez('data/task4_data.npz',
             vertices=vertices,
             faces=faces,
             R1=R1,
             R2=R2,
             R12=R12,
             R21=R21)
    print("\nДанные сохранены в data/task4_data.npz")