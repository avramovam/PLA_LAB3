"""
Лабораторная работа №3: Матрицы в 3D-графике
Задание 1: Создание кубика и других фигур
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

# Создаём папки для данных
os.makedirs('img', exist_ok=True)
os.makedirs('data', exist_ok=True)

# ============================== БАЗОВЫЕ ФУНКЦИИ ==============================

def draw_shape(ax, vertices, faces, face_color='skyblue',
               edge_color='black', alpha=0.8, linewidth=0.8, shade=True):
    """
    Отрисовка фигуры по вершинам и граням
    """
    # Преобразуем однородные координаты в декартовы
    cartesian_vertices = (vertices[:3, :] / vertices[3, :]).T

    # Создаём коллекцию полигонов
    poly_collection = Poly3DCollection(
        cartesian_vertices[faces],
        facecolors=face_color,
        edgecolors=edge_color,
        linewidths=linewidth,
        alpha=alpha,
        shade=shade
    )

    ax.add_collection3d(poly_collection)

    return cartesian_vertices

def create_cube(center=(0, 0, 0), size=2):
    """
    Создание вершин и граней куба
    """
    cx, cy, cz = center
    half = size / 2

    # Вершины куба в однородных координатах (w=1)
    vertices = np.array([
        [cx - half, cx + half, cx + half, cx - half,
         cx - half, cx + half, cx + half, cx - half],  # X
        [cy - half, cy - half, cy + half, cy + half,
         cy - half, cy - half, cy + half, cy + half],  # Y
        [cz - half, cz - half, cz - half, cz - half,
         cz + half, cz + half, cz + half, cz + half],  # Z
        [1, 1, 1, 1, 1, 1, 1, 1]                     # W
    ], dtype=np.float64)

    # Грани куба (индексы вершин)
    faces = np.array([
        [0, 1, 2, 3],  # Нижняя грань
        [4, 5, 6, 7],  # Верхняя грань
        [0, 1, 5, 4],  # Передняя грань
        [2, 3, 7, 6],  # Задняя грань
        [1, 2, 6, 5],  # Правая грань
        [0, 3, 7, 4]   # Левая грань
    ])

    return vertices, faces

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

def create_parallelepiped(center=(0, 0, 0), dimensions=(2, 1.5, 1)):
    """
    Создание параллелепипеда
    """
    cx, cy, cz = center
    dx, dy, dz = dimensions

    # Вершины параллелепипеда
    vertices = np.array([
        [cx - dx/2, cx + dx/2, cx + dx/2, cx - dx/2,
         cx - dx/2, cx + dx/2, cx + dx/2, cx - dx/2],  # X
        [cy - dy/2, cy - dy/2, cy + dy/2, cy + dy/2,
         cy - dy/2, cy - dy/2, cy + dy/2, cy + dy/2],  # Y
        [cz - dz/2, cz - dz/2, cz - dz/2, cz - dz/2,
         cz + dz/2, cz + dz/2, cz + dz/2, cz + dz/2],  # Z
        [1, 1, 1, 1, 1, 1, 1, 1]                     # W
    ], dtype=np.float64)

    # Грани
    faces = np.array([
        [0, 1, 2, 3],  # Нижняя
        [4, 5, 6, 7],  # Верхняя
        [0, 1, 5, 4],  # Передняя
        [2, 3, 7, 6],  # Задняя
        [1, 2, 6, 5],  # Правая
        [0, 3, 7, 4]   # Левая
    ])

    return vertices, faces

# ============================== ЗАДАНИЕ 1 ==============================

def task1():
    """
    Задание 1: Создание кубика и других фигур
    """
    print("="*60)
    print("ЗАДАНИЕ 1: СОЗДАНИЕ КУБИКА И ДРУГИХ ФИГУР")
    print("="*60)

    # 1.1 Создание и отрисовка кубика
    print("\n1.1 Создание и визуализация кубика...")

    vertices_cube, faces_cube = create_cube(center=(0, 0, 0), size=2)

    fig, ax = setup_3d_plot(figsize=(10, 8), elev=25, azim=-45, limits=(-1.5, 1.5), grid=True)

    draw_shape(ax, vertices_cube, faces_cube,
               face_color='lightblue',
               edge_color='navy',
               alpha=0.7,
               linewidth=1.2,
               shade=True)

    ax.set_title("Исходный кубик", fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig('img/task1_cube.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    print("Сохранено: img/task1_cube.png")
    plt.show()

    # 1.2 Создание параллелепипеда
    print("\n1.2 Создание и визуализация параллелепипеда...")

    vertices_par, faces_par = create_parallelepiped(
        center=(0, 0, 0),
        dimensions=(2.5, 1.5, 1.0)
    )

    fig, ax = setup_3d_plot(figsize=(10, 8), elev=30, azim=-50,
                           limits=[(-1.5, 1.5), (-1.0, 1.0), (-0.8, 0.8)], grid=True)

    draw_shape(ax, vertices_par, faces_par,
               face_color='lightgreen',
               edge_color='darkgreen',
               alpha=0.7,
               linewidth=1.2,
               shade=True)

    ax.set_title("Параллелепипед", fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig('img/task1_parallelepiped.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    print("Сохранено: img/task1_parallelepiped.png")
    plt.show()

    # 1.3 Вывод информации
    print("\n1.3 Информация о структуре данных:")
    print(f"   Вершины куба: матрица {vertices_cube.shape}")
    print(f"   Грани куба: матрица {faces_cube.shape}")
    print("\n   Пример вершин (первые 3):")
    for i in range(3):
        print(f"   V{i}: ({vertices_cube[0, i]:+.2f}, {vertices_cube[1, i]:+.2f}, "
              f"{vertices_cube[2, i]:+.2f}, {vertices_cube[3, i]:+.2f})")

    print("\n" + "="*60)
    print("ЗАДАНИЕ 1 ВЫПОЛНЕНО")
    print("="*60)

    return vertices_cube, faces_cube

# ============================== ЗАПУСК ==============================

if __name__ == "__main__":
    vertices_cube, faces_cube = task1()

    # Сохраняем данные для следующих заданий
    np.savez('data/cube_data.npz',
            vertices=vertices_cube,
            faces=faces_cube)
    print("\nДанные сохранены в data/cube_data.npz")