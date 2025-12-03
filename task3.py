"""
Лабораторная работа №3: Матрицы в 3D-графике
Задание 3: Перемещение кубика
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

# Создаём папки для данных
os.makedirs('img', exist_ok=True)


# ============================== ФУНКЦИИ ДЛЯ ЗАДАНИЯ 3 ==============================

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


def draw_shape_simple(ax, vertices, faces, color='lightblue', alpha=0.7, label=None):
    """Упрощённая отрисовка фигуры"""
    cartesian_vertices = (vertices[:3, :] / vertices[3, :]).T
    poly = Poly3DCollection(cartesian_vertices[faces],
                            facecolors=color,
                            edgecolors='darkblue',
                            linewidths=0.8,
                            alpha=alpha,
                            shade=True,
                            label=label)
    ax.add_collection3d(poly)
    return cartesian_vertices


def setup_3d_plot(figsize=(10, 8), elev=25, azim=-45,
                  limits=(-5, 5), grid=True):
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
    """
    Матрица переноса в однородных координатах

    Parameters:
    -----------
    tx, ty, tz : float
        Величины переноса по осям X, Y, Z

    Returns:
    --------
    T : np.ndarray
        Матрица переноса 4x4
    """
    T = np.eye(4, dtype=np.float64)
    T[0, 3] = tx
    T[1, 3] = ty
    T[2, 3] = tz
    return T


def scale_matrix(sx=1.0, sy=1.0, sz=1.0):
    """
    Матрица масштабирования из задания 2
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

def print_theory_translation():
    """Вывод теоретической информации о переносе"""
    print("\n" + "=" * 60)
    print("ТЕОРЕТИЧЕСКАЯ ЧАСТЬ: МАТРИЦА ПЕРЕНОСА")
    print("=" * 60)

    print("\n1. Общая структура матрицы переноса:")
    print("   ⎡ 1  0  0  t_x ⎤")
    print("   ⎢ 0  1  0  t_y ⎥")
    print("   ⎢ 0  0  1  t_z ⎥")
    print("   ⎣ 0  0  0   1  ⎦")

    print("\n2. Как работает преобразование:")
    print("   Для каждой вершины (x, y, z, 1):")
    print("   x' = x + t_x")
    print("   y' = y + t_y")
    print("   z' = z + t_z")
    print("   w' = 1")

    print("\n3. Свойства матрицы переноса:")
    print("   • Единичная подматрица 3x3")
    print("   • Вектор переноса в последнем столбце")
    print("   • Коммутативность: T1 * T2 = T2 * T1")
    print("   • Обратная матрица: T⁻¹(tx,ty,tz) = T(-tx,-ty,-tz)")

    print("\n4. Геометрическая интерпретация:")
    print("   • tx > 0: сдвиг в положительном направлении оси X")
    print("   • tx < 0: сдвиг в отрицательном направлении оси X")
    print("   • Аналогично для ty и tz")


# ============================== ЗАДАНИЕ 3 ==============================

def task3():
    """
    Задание 3: Перемещение кубика
    """
    print("=" * 60)
    print("ЗАДАНИЕ 3: ПЕРЕМЕЩЕНИЕ КУБИКА")
    print("=" * 60)

    # Загружаем данные куба
    print("\n📦 Загрузка данных куба...")
    vertices, faces = load_previous_data()
    print(f"   Загружено: {vertices.shape[1]} вершин, {faces.shape[0]} граней")

    # Вывод теоретической информации
    print_theory_translation()

    # ==================== ЧАСТЬ 1: ОТДЕЛЬНЫЕ ПРЕОБРАЗОВАНИЯ ПЕРЕНОСА ====================
    print("\n" + "=" * 60)
    print("ЧАСТЬ 1: ОТДЕЛЬНЫЕ ПРЕОБРАЗОВАНИЯ ПЕРЕНОСА")
    print("=" * 60)

    # 1. Первое преобразование: перенос по X
    print("\n1. Первое преобразование: перенос по оси X на 3 единицы")
    T1 = translation_matrix(tx=3.0, ty=0.0, tz=0.0)
    print(f"   Матрица T1:\n{T1}")

    vertices_T1 = apply_transformation(vertices, T1)

    fig, ax = setup_3d_plot(figsize=(10, 8), elev=25, azim=-45, limits=(-2, 6), grid=True)

    # Рисуем исходный куб (полупрозрачный)
    draw_shape_simple(ax, vertices, faces, color='lightblue', alpha=0.3, label='Исходный куб')

    # Рисуем перемещённый куб
    draw_shape_simple(ax, vertices_T1, faces, color='lightcoral', alpha=0.7, label='После T1')

    ax.set_title("Перенос: T1 (tx=3, ty=0, tz=0)", fontsize=14, pad=20)
    ax.legend()

    plt.tight_layout()
    plt.savefig('img/task3_translation1.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    print("   Сохранено: img/task3_translation1.png")
    plt.show()

    # 2. Второе преобразование: перенос по Y и Z
    print("\n2. Второе преобразование: перенос по осям Y и Z")
    T2 = translation_matrix(tx=0.0, ty=2.0, tz=1.5)
    print(f"   Матрица T2:\n{T2}")

    vertices_T2 = apply_transformation(vertices, T2)

    fig, ax = setup_3d_plot(figsize=(10, 8), elev=25, azim=-45, limits=(-2, 4), grid=True)

    # Рисуем исходный куб (полупрозрачный)
    draw_shape_simple(ax, vertices, faces, color='lightblue', alpha=0.3, label='Исходный куб')

    # Рисуем перемещённый куб
    draw_shape_simple(ax, vertices_T2, faces, color='lightgreen', alpha=0.7, label='После T2')

    ax.set_title("Перенос: T2 (tx=0, ty=2, tz=1.5)", fontsize=14, pad=20)
    ax.legend()

    plt.tight_layout()
    plt.savefig('img/task3_translation2.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    print("   Сохранено: img/task3_translation2.png")
    plt.show()

    # 3. Комбинированный перенос
    print("\n3. Комбинированное преобразование: T = T2 * T1")
    T_combined = T2 @ T1
    print(f"   Матрица T = T2 * T1:\n{T_combined}")

    vertices_T_combined = apply_transformation(vertices, T_combined)

    fig, ax = setup_3d_plot(figsize=(10, 8), elev=25, azim=-45, limits=(-2, 6), grid=True)

    # Рисуем исходный куб (полупрозрачный)
    draw_shape_simple(ax, vertices, faces, color='lightblue', alpha=0.3, label='Исходный куб')

    # Рисуем перемещённый куб
    draw_shape_simple(ax, vertices_T_combined, faces, color='lightgoldenrodyellow', alpha=0.7, label='После T')

    ax.set_title("Комбинированный перенос: T = T2 * T1", fontsize=14, pad=20)
    ax.legend()

    plt.tight_layout()
    plt.savefig('img/task3_translation_combined.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    print("   Сохранено: img/task3_translation_combined.png")
    plt.show()

    # ==================== ЧАСТЬ 2: КОМБИНАЦИИ ПЕРЕНОСА И МАСШТАБИРОВАНИЯ ====================
    print("\n" + "=" * 60)
    print("ЧАСТЬ 2: КОМБИНАЦИИ ПЕРЕНОСА И МАСШТАБИРОВАНИЯ")
    print("=" * 60)

    # Используем матрицы из задания 2
    S1 = scale_matrix(sx=2.0, sy=1.0, sz=1.0)

    print("\n4. Исследование комбинаций TS и ST:")
    print(f"   Матрица масштабирования S1 (из задания 2):\n{S1}")
    print(f"   Матрица переноса T1:\n{T1}")

    # 4.1 Комбинация TS (сначала перенос, потом масштабирование)
    print("\n   а) TS = T1 * S1 (сначала перенос, потом масштабирование)")
    TS = T1 @ S1
    print(f"   Матрица TS:\n{TS}")

    vertices_TS = apply_transformation(vertices, TS)

    # 4.2 Комбинация ST (сначала масштабирование, потом перенос)
    print("\n   б) ST = S1 * T1 (сначала масштабирование, потом перенос)")
    ST = S1 @ T1
    print(f"   Матрица ST:\n{ST}")

    vertices_ST = apply_transformation(vertices, ST)

    # 4.3 Проверка эквивалентности
    print("\n   в) Проверка эквивалентности TS и ST:")
    print(f"   TS == ST? {np.allclose(TS, ST)}")
    print("   Матрицы не равны, преобразования не эквивалентны!")

    # 4.4 Визуализация обеих комбинаций
    fig = plt.figure(figsize=(12, 5), dpi=100)

    # График TS
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_box_aspect([1, 1, 1])
    ax1.set_xlim(-2, 8)
    ax1.set_ylim(-2, 4)
    ax1.set_zlim(-2, 4)
    ax1.grid(True)
    ax1.view_init(elev=25, azim=-45)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # Исходный куб (полупрозрачный)
    draw_shape_simple(ax1, vertices, faces, color='lightblue', alpha=0.2)

    # Куб после TS
    draw_shape_simple(ax1, vertices_TS, faces, color='lightcoral', alpha=0.7)
    ax1.set_title("TS = T1 * S1\n(сначала перенос, потом масштабирование)", fontsize=12, pad=15)

    # График ST
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_box_aspect([1, 1, 1])
    ax2.set_xlim(-2, 8)
    ax2.set_ylim(-2, 4)
    ax2.set_zlim(-2, 4)
    ax2.grid(True)
    ax2.view_init(elev=25, azim=-45)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    # Исходный куб (полупрозрачный)
    draw_shape_simple(ax2, vertices, faces, color='lightblue', alpha=0.2)

    # Куб после ST
    draw_shape_simple(ax2, vertices_ST, faces, color='lightgreen', alpha=0.7)
    ax2.set_title("ST = S1 * T1\n(сначала масштабирование, потом перенос)", fontsize=12, pad=15)

    plt.tight_layout()
    plt.savefig('img/task3_TS_ST_comparison.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
    print("\n   Сохранено: img/task3_TS_ST_comparison.png")
    plt.show()

    # 4.5 Подробное сравнение одной вершины
    print("\n   г) Подробное сравнение преобразования вершины V0:")
    v0_original = vertices[:, 0]
    print(f"   Исходная вершина V0: {v0_original}")

    # Применяем преобразования
    v0_TS = TS @ v0_original
    v0_ST = ST @ v0_original

    print(f"   После TS: {v0_TS}")
    print(f"   После ST: {v0_ST}")
    print(f"   Разница: {v0_ST - v0_TS}")

    print("\n   д) Геометрическое объяснение:")
    print("   • В TS: сначала перенос, потом масштабирование")
    print("     - Кубик перемещается, затем растягивается")
    print("     - Перенос не масштабируется")
    print("   • В ST: сначала масштабирование, потом перенос")
    print("     - Кубик растягивается, затем перемещается")
    print("     - Компоненты переноса также масштабируются")

    # ==================== ЧАСТЬ 3: ВЫВОДЫ ====================
    print("\n" + "=" * 60)
    print("ВЫВОДЫ")
    print("=" * 60)

    print("\n1. Свойства матрицы переноса:")
    print("   • Матрицы переноса коммутируют между собой")
    print("   • Результирующий перенос = сумме отдельных переносов")
    print("   • T1(tx1,ty1,tz1) * T2(tx2,ty2,tz2) = T(tx1+tx2, ty1+ty2, tz1+tz2)")

    print("\n2. Комбинации переноса и масштабирования:")
    print("   • TS ≠ ST - преобразования не коммутируют")
    print("   • В TS: перенос выполняется до масштабирования")
    print("   • В ST: компоненты переноса также масштабируются")
    print("   • Геометрически: при ST куб перемещается на большее расстояние")

    print("\n3. Геометрическая интерпретация:")
    print("   • T1: сдвиг куба вдоль оси X на 3 единицы")
    print("   • T2: сдвиг куба вдоль осей Y и Z")
    print("   • T: комбинированный сдвиг по всем осям")
    print("   • TS vs ST: разный порядок приводит к разным результатам")

    print("\n4. Практическое значение:")
    print("   • Порядок преобразований в компьютерной графике ВАЖЕН")
    print("   • Обычная последовательность: масштабирование → вращение → перенос")
    print("   • Неправильный порядок приводит к неожиданным результатам")

    print("\n" + "=" * 60)
    print("ЗАДАНИЕ 3 ВЫПОЛНЕНО")
    print("=" * 60)

    return vertices, faces, T1, T2, TS, ST


# ============================== ЗАПУСК ==============================

if __name__ == "__main__":
    # Запускаем задание 3
    vertices, faces, T1, T2, TS, ST = task3()

    # Сохраняем данные для следующих заданий
    np.savez('data/task3_data.npz',
             vertices=vertices,
             faces=faces,
             T1=T1,
             T2=T2,
             TS=TS,
             ST=ST)
    print("\nДанные сохранены в data/task3_data.npz")