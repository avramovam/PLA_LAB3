"""
Лабораторная работа №3: Матрицы в 3D-графике
Задание 5: Вращение кубика вокруг любой вершины
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.linalg import expm
import os

# Создаём папки для данных
os.makedirs('img', exist_ok=True)


# ============================== ФУНКЦИИ ДЛЯ ЗАДАНИЯ 5 ==============================

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


def draw_shape_dual(ax, vertices1, vertices2, faces, color1='lightblue', color2='lightcoral',
                    alpha1=0.3, alpha2=0.7, labels=None):
    """Отрисовка двух фигур (оригинал и преобразованный)"""
    cartesian_vertices1 = (vertices1[:3, :] / vertices1[3, :]).T
    cartesian_vertices2 = (vertices2[:3, :] / vertices2[3, :]).T

    # Оригинал (полупрозрачный)
    poly1 = Poly3DCollection(cartesian_vertices1[faces],
                             facecolors=color1,
                             edgecolors='darkblue',
                             linewidths=0.6,
                             alpha=alpha1,
                             shade=True,
                             label=labels[0] if labels else 'Исходный куб')
    ax.add_collection3d(poly1)

    # Преобразованный
    poly2 = Poly3DCollection(cartesian_vertices2[faces],
                             facecolors=color2,
                             edgecolors='darkred',
                             linewidths=0.8,
                             alpha=alpha2,
                             shade=True,
                             label=labels[1] if labels else 'Повёрнутый куб')
    ax.add_collection3d(poly2)

    return cartesian_vertices1, cartesian_vertices2


def setup_3d_plot(figsize=(10, 8), elev=25, azim=-45,
                  limits=(-3, 3), grid=True):
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


def translation_matrix(tx=0.0, ty=0.0, tz=0.0):
    """Матрица переноса"""
    T = np.eye(4, dtype=np.float64)
    T[0, 3] = tx
    T[1, 3] = ty
    T[2, 3] = tz
    return T


def rotation_matrix_x(theta):
    """Матрица вращения вокруг оси X"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1]
    ], dtype=np.float64)


def rotation_matrix_axis(v, theta):
    """
    Матрица вращения вокруг произвольной оси через начало координат
    """
    v = np.array(v, dtype=np.float64)
    v = v / np.linalg.norm(v)

    vx, vy, vz = v
    J = np.array([
        [0, -vz, vy, 0],
        [vz, 0, -vx, 0],
        [-vy, vx, 0, 0],
        [0, 0, 0, 0]
    ], dtype=np.float64)

    R = expm(J * theta)
    R[3, 3] = 1
    return R


def apply_transformation(vertices, transformation_matrix):
    """Применение матрицы преобразования к вершинам"""
    return transformation_matrix @ vertices


def rotation_around_point(vertices, axis, theta, point):
    """
    Вращение вокруг произвольной точки

    Parameters:
    -----------
    vertices : np.ndarray
        Вершины фигуры
    axis : list or np.ndarray
        Ось вращения
    theta : float
        Угол вращения в радианах
    point : list or np.ndarray
        Точка, вокруг которой происходит вращение

    Returns:
    --------
    rotated_vertices : np.ndarray
        Повёрнутые вершины
    R_total : np.ndarray
        Полная матрица преобразования
    """
    # Шаг 1: Перенос точки вращения в начало координат
    T1 = translation_matrix(-point[0], -point[1], -point[2])

    # Шаг 2: Вращение вокруг начала координат
    R = rotation_matrix_axis(axis, theta)

    # Шаг 3: Обратный перенос
    T2 = translation_matrix(point[0], point[1], point[2])

    # Общая матрица преобразования
    R_total = T2 @ R @ T1

    # Применяем преобразование
    rotated_vertices = apply_transformation(vertices, R_total)

    return rotated_vertices, R_total


# ============================== ТЕОРИЯ И РАСЧЁТЫ ==============================

def print_theory_rotation_around_point():
    """Вывод теоретической информации о вращении вокруг точки"""
    print("\n" + "=" * 60)
    print("ТЕОРЕТИЧЕСКАЯ ЧАСТЬ: ВРАЩЕНИЕ ВОКРУГ ПРОИЗВОЛЬНОЙ ТОЧКИ")
    print("=" * 60)

    print("\n1. Общая формула матрицы вращения вокруг оси v, проходящей через точку M(x,y,z):")
    print("   A = T_M * R_v(θ) * T_{-M}")
    print("\n   где:")
    print("   T_{-M} - матрица переноса точки M в начало координат")
    print("   R_v(θ) - матрица вращения вокруг оси v")
    print("   T_M - обратный перенос из начала координат в точку M")

    print("\n2. Матрицы преобразования:")
    print("\n   а) Перенос в начало координат:")
    print("        ⎡ 1  0  0  -x ⎤")
    print("   T_{-M} = ⎢ 0  1  0  -y ⎥")
    print("        ⎢ 0  0  1  -z ⎥")
    print("        ⎣ 0  0  0   1 ⎦")

    print("\n   б) Обратный перенос:")
    print("        ⎡ 1  0  0  x ⎤")
    print("   T_M = ⎢ 0  1  0  y ⎥")
    print("        ⎢ 0  0  1  z ⎥")
    print("        ⎣ 0  0  0  1 ⎦")

    print("\n3. Геометрическая интерпретация:")
    print("   • Точка M становится новым центром вращения")
    print("   • Преобразование состоит из трёх этапов:")
    print("     1. Перенос: M → (0,0,0)")
    print("     2. Вращение вокруг (0,0,0)")
    print("     3. Обратный перенос: (0,0,0) → M")

    print("\n4. Практическое применение:")
    print("   • Вращение объектов вокруг их центра масс")
    print("   • Вращение вокруг шарниров в анимации")
    print("   • Орбитальное движение в астрономических симуляциях")


# ============================== ЗАДАНИЕ 5 ==============================

def task5():
    """
    Задание 5: Вращение кубика вокруг любой вершины
    """
    print("=" * 60)
    print("ЗАДАНИЕ 5: ВРАЩЕНИЕ КУБИКА ВОКРУГ ЛЮБОЙ ВЕРШИНЫ")
    print("=" * 60)

    # Загружаем данные куба
    print("\n📦 Загрузка данных куба...")
    vertices, faces = load_previous_data()
    print(f"   Загружено: {vertices.shape[1]} вершин, {faces.shape[0]} граней")

    # Вывод теоретической информации
    print_theory_rotation_around_point()

    # ==================== ЧАСТЬ 1: ВЫБОР ВЕРШИНЫ И ПАРАМЕТРОВ ====================
    print("\n" + "=" * 60)
    print("ЧАСТЬ 1: ВЫБОР ВЕРШИНЫ И ПАРАМЕТРОВ ВРАЩЕНИЯ")
    print("=" * 60)

    # Выбираем вершину для вращения (например, вершину V0: (-1, -1, -1))
    vertex_index = 0
    vertex_coords = vertices[:3, vertex_index]
    print(f"\n1. Выбрана вершина V{vertex_index} для вращения:")
    print(f"   Координаты вершины: ({vertex_coords[0]:.1f}, {vertex_coords[1]:.1f}, {vertex_coords[2]:.1f})")

    # Выбираем ось вращения (например, ось X)
    axis = [1, 0, 0]  # Ось X
    theta = np.pi / 2  # Угол 90 градусов

    print(f"\n2. Параметры вращения:")
    print(f"   Ось вращения: v = {axis}")
    print(f"   Угол вращения: θ = π/2 ({np.degrees(theta):.0f}°)")

    # ==================== ЧАСТЬ 2: РАСЧЁТ МАТРИЦЫ ПРЕОБРАЗОВАНИЯ ====================
    print("\n" + "=" * 60)
    print("ЧАСТЬ 2: РАСЧЁТ МАТРИЦЫ ПРЕОБРАЗОВАНИЯ")
    print("=" * 60)

    # Вычисляем матрицу вращения вокруг вершины
    rotated_vertices, R_total = rotation_around_point(vertices, axis, theta, vertex_coords)

    print("\n3. Вычисление матриц преобразования:")

    # Матрица переноса в начало координат
    T1 = translation_matrix(-vertex_coords[0], -vertex_coords[1], -vertex_coords[2])
    print(f"\n   а) Матрица переноса T_{-vertex_coords}:")
    print(f"   T1 = T(-{vertex_coords[0]}, -{vertex_coords[1]}, -{vertex_coords[2]})")
    print(f"   {T1}")

    # Матрица вращения вокруг оси X
    R = rotation_matrix_x(theta)
    print(f"\n   б) Матрица вращения вокруг оси X на π/2:")
    print(f"   R = R_x(π/2)")
    print(f"   {R}")

    # Обратный перенос
    T2 = translation_matrix(vertex_coords[0], vertex_coords[1], vertex_coords[2])
    print(f"\n   в) Обратный перенос T_{vertex_coords}:")
    print(f"   T2 = T({vertex_coords[0]}, {vertex_coords[1]}, {vertex_coords[2]})")
    print(f"   {T2}")

    # Общая матрица
    print(f"\n   г) Общая матрица преобразования A = T2 * R * T1:")
    print(f"   A = {R_total}")

    # Проверка: умножение вручную
    manual_R_total = T2 @ R @ T1
    print(f"\n   д) Проверка: T2 * R * T1 == A? {np.allclose(manual_R_total, R_total)}")

    # ==================== ЧАСТЬ 3: ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ ====================
    print("\n" + "=" * 60)
    print("ЧАСТЬ 3: ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
    print("=" * 60)

    print("\n4. Визуализация вращения вокруг вершины:")

    # Создаём график с исходным и повёрнутым кубом
    fig, ax = setup_3d_plot(figsize=(12, 10), elev=30, azim=-50, limits=(-2, 2), grid=True)

    # Рисуем оба куба
    cart_orig, cart_rot = draw_shape_dual(
        ax, vertices, rotated_vertices, faces,
        color1='lightblue', color2='lightcoral',
        alpha1=0.3, alpha2=0.7,
        labels=['Исходный куб', f'Повёрнут на {np.degrees(theta):.0f}° вокруг V{vertex_index}']
    )

    # Отмечаем точку вращения
    ax.scatter([vertex_coords[0]], [vertex_coords[1]], [vertex_coords[2]],
               color='red', s=100, zorder=10, label=f'Вершина V{vertex_index}')

    # Рисуем ось вращения
    axis_length = 2
    axis_vector = np.array(axis) * axis_length
    ax.quiver(vertex_coords[0], vertex_coords[1], vertex_coords[2],
              axis_vector[0], axis_vector[1], axis_vector[2],
              color='darkred', linewidth=2, arrow_length_ratio=0.1,
              label=f'Ось вращения: {axis}')

    # Настраиваем график
    ax.set_title(f"Вращение куба вокруг вершины V{vertex_index} на {np.degrees(theta):.0f}°",
                 fontsize=14, pad=20)
    ax.legend(loc='upper left')

    # Добавляем информационную панель
    info_text = (f"Параметры вращения:\n"
                 f"• Вершина: V{vertex_index} ({vertex_coords[0]:.1f}, {vertex_coords[1]:.1f}, {vertex_coords[2]:.1f})\n"
                 f"• Ось: {axis}\n"
                 f"• Угол: {np.degrees(theta):.0f}°\n"
                 f"• Матрица: A = T2 * R * T1")

    plt.figtext(0.02, 0.02, info_text, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    plt.savefig('img/task5_rotation_around_vertex.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    print(f"   Сохранено: img/task5_rotation_around_vertex.png")
    plt.show()

    # ==================== ЧАСТЬ 4: ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ ====================
    print("\n" + "=" * 60)
    print("ЧАСТЬ 4: ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ")
    print("=" * 60)

    print("\n5. Анализ результатов преобразования:")

    # Анализ преобразования выбранной вершины
    print(f"\n   а) Преобразование вершины V{vertex_index}:")
    v_original = vertices[:, vertex_index]
    v_rotated = rotated_vertices[:, vertex_index]
    print(f"   Исходные координаты: ({v_original[0]:.3f}, {v_original[1]:.3f}, {v_original[2]:.3f})")
    print(f"   После вращения: ({v_rotated[0]:.3f}, {v_rotated[1]:.3f}, {v_rotated[2]:.3f})")
    print(f"   Вершина осталась на месте? {np.allclose(v_original[:3], v_rotated[:3], atol=1e-10)}")

    # Анализ другой вершины (например, противоположной)
    opposite_index = 6  # Противоположная вершина
    v_opp_original = vertices[:3, opposite_index]
    v_opp_rotated = rotated_vertices[:3, opposite_index]

    print(f"\n   б) Преобразование противоположной вершины V{opposite_index}:")
    print(f"   Исходные координаты: ({v_opp_original[0]:.3f}, {v_opp_original[1]:.3f}, {v_opp_original[2]:.3f})")
    print(f"   После вращения: ({v_opp_rotated[0]:.3f}, {v_opp_rotated[1]:.3f}, {v_opp_rotated[2]:.3f})")

    # Вычисляем смещение
    displacement = v_opp_rotated - v_opp_original
    distance = np.linalg.norm(displacement)
    print(f"   Смещение: ({displacement[0]:.3f}, {displacement[1]:.3f}, {displacement[2]:.3f})")
    print(f"   Расстояние смещения: {distance:.3f}")

    # Проверка сохранения расстояний
    print(f"\n   в) Проверка сохранения расстояний:")
    orig_distances = []
    rot_distances = []

    for i in range(vertices.shape[1]):
        dist_orig = np.linalg.norm(vertices[:3, i] - vertices[:3, vertex_index])
        dist_rot = np.linalg.norm(rotated_vertices[:3, i] - rotated_vertices[:3, vertex_index])
        orig_distances.append(dist_orig)
        rot_distances.append(dist_rot)

    print(f"   Расстояния от V{vertex_index} до других вершин (до и после):")
    for i in range(len(orig_distances)):
        print(f"   V{i}: {orig_distances[i]:.3f} → {rot_distances[i]:.3f} "
              f"(совпадение: {np.isclose(orig_distances[i], rot_distances[i])})")

    # ==================== ЧАСТЬ 5: ВЫВОДЫ ====================
    print("\n" + "=" * 60)
    print("ВЫВОДЫ")
    print("=" * 60)

    print("\n1. Результаты преобразования:")
    print("   • Выбранная вершина остаётся неподвижной")
    print("   • Все остальные вершины вращаются вокруг этой вершины")
    print("   • Расстояния от центра вращения сохраняются")
    print("   • Форма куба сохраняется (ортогональное преобразование)")

    print("\n2. Математические аспекты:")
    print("   • Общая матрица: A = T_M * R_v(θ) * T_{-M}")
    print("   • Преобразование состоит из трёх этапов")
    print("   • Матрица A также является ортогональной (сохраняет расстояния)")

    print("\n3. Геометрическая интерпретация:")
    print("   • Точка M становится новым центром вращения")
    print("   • Преобразование можно представить как:")
    print("     1. Перенос системы координат в точку M")
    print("     2. Вращение в новой системе координат")
    print("     3. Обратный перенос в исходную систему")

    print("\n4. Практическое применение:")
    print("   • Анимация вращения объектов вокруг шарниров")
    print("   • Вращение планет вокруг собственной оси")
    print("   • Преобразования в робототехнике (вращение вокруг точек крепления)")

    print("\n5. Особенности реализации:")
    print("   • Использованы однородные координаты")
    print("   • Все преобразования представлены матрицами 4×4")
    print("   • Порядок умножения матриц соответствует порядку преобразований")

    print("\n" + "=" * 60)
    print("ЗАДАНИЕ 5 ВЫПОЛНЕНО")
    print("=" * 60)

    return vertices, faces, rotated_vertices, R_total, vertex_coords


# ============================== ЗАПУСК ==============================

if __name__ == "__main__":
    # Запускаем задание 5
    vertices, faces, rotated_vertices, R_total, vertex_coords = task5()

    # Сохраняем данные для следующих заданий
    np.savez('data/task5_data.npz',
             vertices=vertices,
             faces=faces,
             rotated_vertices=rotated_vertices,
             R_total=R_total,
             vertex_coords=vertex_coords)
    print("\nДанные сохранены в data/task5_data.npz")