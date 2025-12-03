"""
Лабораторная работа №3: Матрицы в 3D-графике
Задание 2: Изменение масштаба кубика
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

# Создаём папки для данных
os.makedirs('img', exist_ok=True)

# ============================== ФУНКЦИИ ДЛЯ ЗАДАНИЯ 2 ==============================

def load_cube_data():
    """Загрузка данных куба из задания 1"""
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

def draw_shape_simple(ax, vertices, faces, color='lightblue', alpha=0.7):
    """Упрощённая отрисовка фигуры"""
    cartesian_vertices = (vertices[:3, :] / vertices[3, :]).T
    ax.add_collection3d(Poly3DCollection(cartesian_vertices[faces],
                                         facecolors=color,
                                         edgecolors='darkblue',
                                         linewidths=0.8,
                                         alpha=alpha,
                                         shade=True))
    return cartesian_vertices

def setup_3d_plot(figsize=(10, 8), elev=25, azim=-45,
                  limits=(-2, 2), grid=True):
    """
    Настройка 3D графика с координатной сеткой, но без осей
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

def scale_matrix(sx=1.0, sy=1.0, sz=1.0):
    """
    Матрица масштабирования в однородных координатах
    """
    S = np.eye(4, dtype=np.float64)
    S[0, 0] = sx
    S[1, 1] = sy
    S[2, 2] = sz
    return S

def apply_transformation(vertices, transformation_matrix):
    """
    Применение матрицы преобразования к вершинам
    """
    return transformation_matrix @ vertices

# ============================== ТЕОРИЯ И РАСЧЁТЫ ==============================

def print_theory():
    """Вывод теоретической информации"""
    print("\n" + "="*60)
    print("ТЕОРЕТИЧЕСКАЯ ЧАСТЬ")
    print("="*60)

    print("\n1. Общая структура матрицы масштабирования:")
    print("   ⎡ s_x   0    0    0 ⎤")
    print("   ⎢  0   s_y   0    0 ⎥")
    print("   ⎢  0    0   s_z   0 ⎥")
    print("   ⎣  0    0    0    1 ⎦")

    print("\n2. Свойства матрицы масштабирования:")
    print("   • Диагональная матрица")
    print("   • Коммутативность: S1 * S2 = S2 * S1")
    print("   • Обратная матрица: S⁻¹ = diag(1/sx, 1/sy, 1/sz, 1)")

    print("\n3. Как работает преобразование:")
    print("   Для каждой вершины (x, y, z, 1):")
    print("   x' = s_x * x")
    print("   y' = s_y * y")
    print("   z' = s_z * z")
    print("   w' = 1")

    print("\n4. Геометрическая интерпретация:")
    print("   • s_x > 1: растяжение по оси X")
    print("   • s_x < 1: сжатие по оси X")
    print("   • s_x = 1: сохранение размера по оси X")
    print("   • s_x < 0: отражение относительно плоскости YZ")

# ============================== ЗАДАНИЕ 2 ==============================

def task2():
    """
    Задание 2: Изменение масштаба кубика
    """
    print("="*60)
    print("ЗАДАНИЕ 2: ИЗМЕНЕНИЕ МАСШТАБА КУБИКА")
    print("="*60)

    # Загружаем данные куба
    print("\n📦 Загрузка данных куба из задания 1...")
    vertices, faces = load_cube_data()
    print(f"   Загружено: {vertices.shape[1]} вершин, {faces.shape[0]} граней")

    # Вывод теоретической информации
    print_theory()

    # ==================== ЧАСТЬ 1: ОТДЕЛЬНЫЕ ПРЕОБРАЗОВАНИЯ ====================
    print("\n" + "="*60)
    print("ЧАСТЬ 1: ОТДЕЛЬНЫЕ ПРЕОБРАЗОВАНИЯ МАСШТАБИРОВАНИЯ")
    print("="*60)

    # 1. Первое преобразование: растяжение по X
    print("\n1. Первое преобразование: растяжение по оси X в 2 раза")
    S1 = scale_matrix(sx=2.0, sy=1.0, sz=1.0)
    print(f"   Матрица S1:\n{S1}")

    vertices_S1 = apply_transformation(vertices, S1)

    fig, ax = setup_3d_plot(figsize=(8, 6), elev=25, azim=-45, limits=(-2.5, 2.5), grid=True)
    draw_shape_simple(ax, vertices_S1, faces, color='lightcoral', alpha=0.7)
    ax.set_title("Масштабирование: S1 (sx=2, sy=1, sz=1)", fontsize=12, pad=15)

    plt.tight_layout()
    plt.savefig('img/task2_scale1.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    print("   Сохранено: img/task2_scale1.png")
    plt.show()

    # 2. Второе преобразование: сжатие по Y и Z
    print("\n2. Второе преобразование: сжатие по осям Y и Z в 2 раза")
    S2 = scale_matrix(sx=1.0, sy=0.5, sz=0.5)
    print(f"   Матрица S2:\n{S2}")

    vertices_S2 = apply_transformation(vertices, S2)

    fig, ax = setup_3d_plot(figsize=(8, 6), elev=25, azim=-45, limits=(-1.5, 1.5), grid=True)
    draw_shape_simple(ax, vertices_S2, faces, color='lightgreen', alpha=0.7)
    ax.set_title("Масштабирование: S2 (sx=1, sy=0.5, sz=0.5)", fontsize=12, pad=15)

    plt.tight_layout()
    plt.savefig('img/task2_scale2.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    print("   Сохранено: img/task2_scale2.png")
    plt.show()

    # ==================== ЧАСТЬ 2: КОМБИНИРОВАННОЕ ПРЕОБРАЗОВАНИЕ ====================
    print("\n" + "="*60)
    print("ЧАСТЬ 2: КОМБИНИРОВАННОЕ ПРЕОБРАЗОВАНИЕ")
    print("="*60)

    print("\n3. Комбинированное преобразование: S = S2 * S1")
    S_combined = S2 @ S1
    print(f"   Матрица S = S2 * S1:\n{S_combined}")

    print("\n   Проверка коммутативности:")
    print(f"   S1 * S2:\n{S1 @ S2}")
    print(f"   S2 * S1:\n{S2 @ S1}")
    print("   Матрицы равны? ", np.allclose(S1 @ S2, S2 @ S1))

    vertices_S_combined = apply_transformation(vertices, S_combined)

    fig, ax = setup_3d_plot(figsize=(8, 6), elev=25, azim=-45, limits=(-2.5, 2.5), grid=True)
    draw_shape_simple(ax, vertices_S_combined, faces, color='lightgoldenrodyellow', alpha=0.7)
    ax.set_title("Комбинированное масштабирование: S = S2 * S1", fontsize=12, pad=15)

    plt.tight_layout()
    plt.savefig('img/task2_scale_combined.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    print("   Сохранено: img/task2_scale_combined.png")
    plt.show()

    # ==================== ЧАСТЬ 3: СРАВНЕНИЕ ВСЕХ ПРЕОБРАЗОВАНИЙ ====================
    print("\n" + "="*60)
    print("ЧАСТЬ 3: СРАВНЕНИЕ ВСЕХ ПРЕОБРАЗОВАНИЙ")
    print("="*60)

    fig = plt.figure(figsize=(15, 5), dpi=100)

    # Исходный куб
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_box_aspect([1, 1, 1])
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_zlim(-1.5, 1.5)
    ax1.grid(True)
    ax1.view_init(elev=25, azim=-45)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    draw_shape_simple(ax1, vertices, faces, color='lightblue', alpha=0.7)
    ax1.set_title("Исходный куб", fontsize=12, pad=15)

    # Преобразование S1
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.set_box_aspect([1, 1, 1])
    ax2.set_xlim(-2.5, 2.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_zlim(-1.5, 1.5)
    ax2.grid(True)
    ax2.view_init(elev=25, azim=-45)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    draw_shape_simple(ax2, vertices_S1, faces, color='lightcoral', alpha=0.7)
    ax2.set_title("S1 (sx=2)", fontsize=12, pad=15)

    # Преобразование S_combined
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.set_box_aspect([1, 1, 1])
    ax3.set_xlim(-2.5, 2.5)
    ax3.set_ylim(-1.0, 1.0)
    ax3.set_zlim(-1.0, 1.0)
    ax3.grid(True)
    ax3.view_init(elev=25, azim=-45)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    draw_shape_simple(ax3, vertices_S_combined, faces, color='lightgoldenrodyellow', alpha=0.7)
    ax3.set_title("S = S2 * S1", fontsize=12, pad=15)

    plt.tight_layout()
    plt.savefig('img/task2_comparison.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    print("\n   Сохранено: img/task2_comparison.png")
    plt.show()

    # ==================== ЧАСТЬ 4: ВЫВОДЫ ====================
    print("\n" + "="*60)
    print("ВЫВОДЫ")
    print("="*60)

    print("\n1. Результаты преобразований:")
    print(f"   Исходный куб: размеры 2×2×2")
    print(f"   После S1: размеры {2*2}×{2}×{2} = 4×2×2")
    print(f"   После S2: размеры {2}×{2*0.5}×{2*0.5} = 2×1×1")
    print(f"   После S = S2 * S1: размеры {2*2}×{2*0.5}×{2*0.5} = 4×1×1")

    print("\n2. Наблюдения:")
    print("   • Матрицы масштабирования коммутируют (S1*S2 = S2*S1)")
    print("   • Комбинированное преобразование эквивалентно")
    print("     последовательному применению отдельных преобразований")
    print("   • Порядок применения не важен для масштабирования")

    print("\n3. Геометрическая интерпретация:")
    print("   • S1: растяжение в 2 раза по оси X")
    print("   • S2: сжатие в 2 раза по осям Y и Z")
    print("   • S: одновременно растяжение по X и сжатие по Y и Z")

    print("\n" + "="*60)
    print("ЗАДАНИЕ 2 ВЫПОЛНЕНО")
    print("="*60)

    return vertices, faces, S1, S2, S_combined

# ============================== ЗАПУСК ==============================

if __name__ == "__main__":
    # Запускаем задание 2
    vertices, faces, S1, S2, S_combined = task2()

    # Сохраняем данные для следующих заданий
    np.savez('data/task2_data.npz',
            vertices=vertices,
            faces=faces,
            S1=S1,
            S2=S2,
            S_combined=S_combined)
    print("\nДанные сохранены в data/task2_data.npz")