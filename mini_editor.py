import pygame
import moderngl
import numpy as np
from pyrr import Matrix44, Vector3, Quaternion
import math

# ... (Классы Camera, Object3D, GeometryFactory остаются без изменений) ...

class Camera:
    def __init__(self, position=None, target=None):
        self.target = target or Vector3([0.0, 0.0, 0.0])
        self.up = Vector3([0.0, 1.0, 0.0])
        self.distance = 10.0
        self.azimuth = 0.0
        # Начальный наклон чуть сверху
        self.elevation = math.pi / 6 
        self.fov = 45.0
        self.near = 0.1
        self.far = 100.0
        self.aspect = 1.0
        self._update_position()
    
    def _update_position(self):
        # Математика пересчета углов в координаты
        x = self.distance * math.cos(self.elevation) * math.sin(self.azimuth)
        y = self.distance * math.sin(self.elevation)
        z = self.distance * math.cos(self.elevation) * math.cos(self.azimuth)
        self.position = self.target + Vector3([x, y, z])
    
    def rotate(self, delta_azimuth, delta_elevation):
        # Вращение без инверсии: куда ведем мышь, туда и камера
        self.azimuth -= delta_azimuth
        # Ограничение, чтобы не перевернуться через зенит
        limit = math.pi / 2 - 0.05
        self.elevation = max(-limit, min(limit, self.elevation + delta_elevation))
        self._update_position()

    def zoom(self, delta):
        # Приближение должно быть пропорционально дистанции (плавнее)
        self.distance = max(0.5, self.distance + delta * (self.distance * 0.1))
        self._update_position()

    def pan(self, dx, dy):
        # ГЛАВНОЕ ИСПРАВЛЕНИЕ: Движение относительно экрана
        # 1. Вычисляем вектор взгляда (Forward)
        forward = self.target - self.position
        forward_norm = forward / np.linalg.norm(forward)
        
        # 2. Вычисляем вектор "Вправо" (Right) через векторное произведение
        right = np.cross(forward_norm, [0, 1, 0])
        right_norm = right / np.linalg.norm(right)
        
        # 3. Вычисляем вектор "Вверх" (Up) относительно камеры
        up_dir = np.cross(right_norm, forward_norm)
        
        # Коэффициент скорости зависит от дистанции
        speed = self.distance * 0.001
        
        # Сдвигаем цель и позицию по осям экрана
        self.target -= right_norm * dx * speed
        self.target += up_dir * dy * speed
        self._update_position()

    def get_view_matrix(self):
        return Matrix44.look_at(self.position, self.target, self.up)

    def get_projection_matrix(self):
        return Matrix44.perspective_projection(self.fov, self.aspect, self.near, self.far)


class Object3D:
    def __init__(self, vertices, indices, name="Object", color=(1.0, 1.0, 1.0)):
        # Порядок изменен: сначала данные, потом имя
        self.name = name
        self.vertices = np.array(vertices, dtype='f4')
        self.indices = np.array(indices, dtype='i4')
        self.color = color
        
        self.position = Vector3([0.0, 0.0, 0.0])
        # Инициализируем кватернион явно, чтобы избежать ошибок
        self.rotation = Quaternion([0.0, 0.0, 0.0, 1.0])
        self.scale = Vector3([1.0, 1.0, 1.0])
        self.selected = False
        
        self.vbo = None
        self.ibo = None
        self.vao = None
        
        # Если у тебя есть метод _compute_bounds, оставляем его
        if hasattr(self, '_compute_bounds'):
            self._compute_bounds()

    def get_model_matrix(self):
        # ЗАЩИТА: Проверяем на NaN (Not a Number). 
        # Если вращение сломалось, сбрасываем его в стандартное положение.
        if np.isnan(np.sum(self.rotation)):
            self.rotation = Quaternion([0.0, 0.0, 0.0, 1.0])
            print(f"Warning: Rotation reset for {self.name}")

        m = Matrix44.from_translation(self.position)
        m *= Matrix44.from_quaternion(self.rotation)
        m *= Matrix44.from_scale(self.scale)
        return m
    
    def _compute_bounds(self):
        verts = self.vertices.reshape(-1, 3)
        self.bounds_min = np.min(verts, axis=0)
        self.bounds_max = np.max(verts, axis=0)
        self.bounds_center = (self.bounds_min + self.bounds_max) / 2
        self.bounds_radius = np.linalg.norm(self.bounds_max - self.bounds_min) / 2
    
    def get_model_matrix(self):
        translation = Matrix44.from_translation(self.position)
        rotation = Matrix44.from_quaternion(self.rotation)
        scale = Matrix44.from_scale(self.scale)
        return translation * rotation * scale
    
    def get_world_position(self):
        return self.position + self.bounds_center * self.scale
    
    def translate(self, offset):
        self.position += offset
    
    def rotate_euler(self, euler_angles):
        qx = Quaternion.from_x_rotation(euler_angles[0])
        qy = Quaternion.from_y_rotation(euler_angles[1])
        qz = Quaternion.from_z_rotation(euler_angles[2])
        self.rotation = qz * qy * qx * self.rotation
    
    def set_scale(self, scale):
        if isinstance(scale, (int, float)):
            self.scale = Vector3([scale, scale, scale])
        else:
            self.scale = Vector3(scale)

class GeometryFactory:
    @staticmethod
    def create_cube():
        v = np.array([-1,-1,1, 1,-1,1, 1,1,1, -1,1,1, -1,-1,-1, 1,-1,-1, 1,1,-1, -1,1,-1], 'f4')
        i = np.array([0,1,2, 2,3,0, 1,5,6, 6,2,1, 7,6,5, 5,4,7, 4,0,3, 3,7,4, 3,2,6, 6,7,3, 4,5,1, 1,0,4], 'i4')
        return Object3D(v, i, "Cube")

    @staticmethod

    def create_sphere(segments=16):

        v, idx = [], []

        for i in range(segments + 1):

            lat = math.pi * i / segments - math.pi/2

            for j in range(segments + 1):

                lon = 2 * math.pi * j / segments

                v.extend([math.cos(lat)*math.cos(lon), math.sin(lat), math.cos(lat)*math.sin(lon)])

        for i in range(segments):

            for j in range(segments):

                p1, p2 = i*(segments+1)+j, (i+1)*(segments+1)+j

                idx.extend([p1, p2, p1+1, p1+1, p2, p2+1])

        return Object3D(np.array(v, 'f4'), np.array(idx, 'i4'), "Sphere")
    @staticmethod
    def create_pyramid():
        v = np.array([0,1,0, -1,-1,1, 1,-1,1, 1,-1,-1, -1,-1,-1], 'f4')
        i = np.array([0,1,2, 0,2,3, 0,3,4, 0,4,1, 1,4,3, 3,2,1], 'i4')
        return Object3D(v, i, "Pyramid")

    @staticmethod
    def create_plane():
        v = np.array([-2,0,-2, 2,0,-2, 2,0,2, -2,0,2], 'f4')
        i = np.array([0,1,2, 2,3,0], 'i4')
        return Object3D(v, i, "Plane")

    @staticmethod
    def create_cylinder(segments=16):
        v, idx = [], []
        # Боковая поверхность
        for i in range(segments + 1):
            angle = 2 * math.pi * i / segments
            x, z = math.cos(angle), math.sin(angle)
            v.extend([x, 1, z, x, -1, z])
            if i < segments:
                p = i * 2
                idx.extend([p, p+1, p+2, p+2, p+1, p+3])
        
        # Крышки
        top_center = len(v) // 3
        v.extend([0, 1, 0])
        bot_center = len(v) // 3
        v.extend([0, -1, 0])
        
        for i in range(segments):
            p = i * 2
            idx.extend([top_center, p, p + 2])
            idx.extend([bot_center, p + 1, p + 3])
        return Object3D(np.array(v, 'f4'), np.array(idx, 'i4'), "Cylinder")

    @staticmethod
    def create_cone(segments=16, rings=8):
        v, idx = [], []
        # Генерация капсулы через сферические координаты для куполов
        for i in range(rings * 2 + 1):
            y_offset = 1.0 if i <= rings else -1.0
            phi = math.pi * 0.5 * (i / rings)
            if i > rings: phi = math.pi * 0.5 + math.pi * 0.5 * ((i-rings)/rings)
            
            lat = math.pi/2 - phi
            curr_y = math.sin(lat) + y_offset
            radius = math.cos(lat)
            
            for j in range(segments + 1):
                lon = 2 * math.pi * j / segments
                v.extend([radius * math.cos(lon), curr_y * 0.5, radius * math.sin(lon)])
                
        for i in range(rings * 2):
            for j in range(segments):
                p1 = i * (segments + 1) + j
                p2 = p1 + segments + 1
                idx.extend([p1, p2, p1+1, p1+1, p2, p2+1])
        return Object3D(v, idx, "Capsule")

    @staticmethod
    def create_torus(segments=16, ring_segments=8, r1=1.0, r2=0.3):
        v, idx = [], []
        for i in range(segments + 1):
            u = 2 * math.pi * i / segments
            for j in range(ring_segments + 1):
                v_angle = 2 * math.pi * j / ring_segments
                x = (r1 + r2 * math.cos(v_angle)) * math.cos(u)
                y = r2 * math.sin(v_angle)
                z = (r1 + r2 * math.cos(v_angle)) * math.sin(u)
                v.extend([x, y, z])
        for i in range(segments):
            for j in range(ring_segments):
                p1 = i * (ring_segments + 1) + j
                p2 = p1 + ring_segments + 1
                idx.extend([p1, p2, p1+1, p1+1, p2, p2+1])
        return Object3D(np.array(v, 'f4'), np.array(idx, 'i4'), "Torus")

    @staticmethod
    def create_capsule(segments=24, rings=12):
        # Капсула Unity (Клавиша 8)
        radius, cylinder_h = 0.5, 1.0 
        v, idx = [], []
        for i in range(rings * 2 + 1):
            y_off = 0.5 if i <= rings else -0.5
            phi = math.pi * (i / (rings * 2))
            lat = math.pi/2 - phi
            curr_y = radius * math.sin(lat) + y_off
            curr_r = radius * math.cos(lat)
            for j in range(segments + 1):
                lon = 2 * math.pi * j / segments
                v.extend([curr_r * math.cos(lon), curr_y, curr_r * math.sin(lon)])
        for i in range(rings * 2):
            for j in range(segments):
                p1, p2 = i*(segments+1)+j, (i+1)*(segments+1)+j
                idx.extend([p1, p2, p1+1, p1+1, p2, p2+1])
        return Object3D(np.array(v, 'f4'), np.array(idx, 'i4'), "Capsule")
class Scene:
    def __init__(self, ctx):
        self.ctx = ctx
        self.objects = []
        self.selected_object = None
        self.history = []
        self.max_undo = 50 # Чтобы не забивать память бесконечно
        # ВАЖНО: Сначала создаем программу, потом всё остальное
        self.program = self._create_shader_program()
        self._create_grid()

    def export_to_obj(self, filename="scene_export.obj"):
        try:
            with open(filename, "w") as f:
                f.write("# Exported from Python Mini Editor\n")
                vertex_offset = 1 

                # ИСПРАВЛЕНО: просто self.objects, а не self.scene.objects
                for obj in self.objects:
                    f.write(f"o {obj.name}\n")
                    model_mat = obj.get_model_matrix()
                    
                    # Вершины
                    for i in range(0, len(obj.vertices), 3):
                        local_v = Vector3([obj.vertices[i], obj.vertices[i+1], obj.vertices[i+2]])
                        world_v = model_mat * local_v
                        f.write(f"v {world_v.x:.4f} {world_v.y:.4f} {world_v.z:.4f}\n")
                    
                    # Грани
                    for i in range(0, len(obj.indices), 3):
                        i1 = obj.indices[i] + vertex_offset
                        i2 = obj.indices[i+1] + vertex_offset
                        i3 = obj.indices[i+2] + vertex_offset
                        f.write(f"f {i1} {i2} {i3}\n")
                    
                    vertex_offset += len(obj.vertices) // 3
                    
            print(f"Сцена успешно сохранена в {filename}")
        except Exception as e:
            print(f"Ошибка при экспорте: {e}")

    def _create_shader_program(self):
        vs = '''#version 330
        in vec3 in_position;
        uniform mat4 model; uniform mat4 view; uniform mat4 projection;
        out vec3 v_pos;
        void main() {
            v_pos = (model * vec4(in_position, 1.0)).xyz;
            gl_Position = projection * view * model * vec4(in_position, 1.0);
        }'''
        
        fs = '''#version 330
        in vec3 v_pos;
        uniform vec3 color; uniform bool selected; uniform vec3 viewPos;
        out vec4 fragColor;
        void main() {
            // Геометрическая нормаль для четких граней
            vec3 N = normalize(cross(dFdx(v_pos), dFdy(v_pos)));
            vec3 L = normalize(viewPos - v_pos); // Свет "из глаз"
            
            // Расчет освещения (смесь прямого и мягкого заполняющего)
            float diff = max(dot(N, L), 0.0) * 0.7 + 0.3;
            vec3 res = color * diff;
            
            if (selected) {
                res = mix(res, vec3(1.0, 0.6, 0.0), 0.4); // Оранжевый контур как в Blender
                res += 0.2;
            }
            fragColor = vec4(res, 1.0);
        }'''
        return self.ctx.program(vertex_shader=vs, fragment_shader=fs)

    def _create_grid(self):
        # Оставляем логику сетки как была, она теперь увидит self.program
        grid_size = 20
        lines = []
        for i in range(-grid_size, grid_size + 1):
            lines.extend([float(i), 0, float(-grid_size), float(i), 0, float(grid_size)])
            lines.extend([float(-grid_size), 0, float(i), float(grid_size), 0, float(i)])
        grid_data = np.array(lines, dtype='f4')
        self.grid_vbo = self.ctx.buffer(grid_data.tobytes())
        self.grid_vao = self.ctx.vertex_array(self.program, [(self.grid_vbo, '3f', 'in_position')])

    def add_object(self, obj):
        self.objects.append(obj)

    def select_object(self, obj):
        self.selected_object = obj

    def remove_selected(self):
        if self.selected_object in self.objects:
            self.objects.remove(self.selected_object)
            self.selected_object = None

    def render(self, camera):
        # Очистка контекста (если нужно принудительно, хотя она есть в Editor3D.run)
        view_mat = np.array(camera.get_view_matrix(), dtype='f4')
        proj_mat = np.array(camera.get_projection_matrix(), dtype='f4')
        view_pos = np.array(camera.position, dtype='f4')
        
        # Передаем матрицы
        self.program['view'].write(view_mat.tobytes())
        self.program['projection'].write(proj_mat.tobytes())
        
        # Безопасная запись viewPos (проверяем наличие в шейдере)
        if 'viewPos' in self.program:
            self.program['viewPos'].write(view_pos.tobytes())
        
        # Сетка
        self.program['model'].write(np.array(Matrix44.identity(), dtype='f4').tobytes())
        self.program['color'].value = (0.3, 0.3, 0.3)
        self.program['selected'].value = False
        self.grid_vao.render(moderngl.LINES)
        
        # Объекты
        # Внутри Scene.render, цикл отрисовки объектов:
        for obj in self.objects:
            if obj.vao is None:
                obj.vbo = self.ctx.buffer(obj.vertices.tobytes())
                obj.ibo = self.ctx.buffer(obj.indices.tobytes())
                # Передаем только позицию '3f'
                obj.vao = self.ctx.vertex_array(
                    self.program, 
                    [(obj.vbo, '3f', 'in_position')], 
                    obj.ibo
                )

            model_mat = np.array(obj.get_model_matrix(), dtype='f4')
            self.program['model'].write(model_mat.tobytes())
            self.program['color'].value = obj.color
            self.program['selected'].value = (obj == self.selected_object)
            if 'viewPos' in self.program:
                self.program['viewPos'].write(np.array(camera.position, dtype='f4').tobytes())
            obj.vao.render()

# ... (Остальной код InputController и Editor3D остается без изменений) ...

class InputController:
    def __init__(self, scene, camera):
        self.scene = scene
        self.camera = camera
        self.mouse_pressed = {}
        self.transform_mode = None   # 'G', 'R', 'S'
        self.transform_axis = None   # 'X', 'Y', 'Z'
        self.transform_space = 'GLOBAL'

    def handle_event(self, event):
        keys = pygame.key.get_pressed()
        alt_pressed = keys[pygame.K_LALT] or keys[pygame.K_RALT]
        # В InputController.handle_event
        if event.type == pygame.MOUSEMOTION:
            rel_x, rel_y = event.rel
            if not self.transform_mode:
                if alt_pressed and self.mouse_pressed.get(1):
                    # Передаем как есть, внутри Camera.rotate мы все настроили
                    self.camera.rotate(rel_x * 0.005, rel_y * 0.005)
            
            # ЭФФЕКТ BLENDER: Зацикливание мышки при трансформации
            if self.transform_mode:
                width, height = pygame.display.get_surface().get_size()
                mouse_x, mouse_y = event.pos
                margin = 10 # Порог срабатывания у края
                
                new_x, new_y = mouse_x, mouse_y
                changed = False

                # Проверяем границы по горизонтали
                if mouse_x <= 0: 
                    new_x = width - margin; changed = True
                elif mouse_x >= width - 1: 
                    new_x = margin; changed = True

                # Проверяем границы по вертикали
                if mouse_y <= 0: 
                    new_y = height - margin; changed = True
                elif mouse_y >= height - 1: 
                    new_y = margin; changed = True

                if changed:
                    pygame.mouse.set_pos(new_x, new_y)
                    # Важно: после set_pos следующий mousemotion даст огромный rel_x/y.
                    # Но в режиме трансформации мы уже получили rel_x/y текущего кадра,
                    # поэтому телепортация не испортит движение объекта.

            # Обычная логика движения
            if not self.transform_mode:
                if alt_pressed and self.mouse_pressed.get(1):
                    self.camera.rotate(rel_x * 0.005, rel_y * 0.005)
                elif alt_pressed and self.mouse_pressed.get(3):
                    self.camera.pan(rel_x, rel_y)
            else:
                self._apply_motion(rel_x, rel_y)

        if event.type == pygame.MOUSEMOTION:
            rel_x, rel_y = event.rel
            if not self.transform_mode:
                # Вращение камеры (Alt + ЛКМ) - теперь знаки правильные
                if alt_pressed and self.mouse_pressed.get(1):
                    self.camera.rotate(rel_x * 0.005, -rel_y * 0.005)
                # Панорамирование (Alt + ПКМ)
                elif alt_pressed and self.mouse_pressed.get(3):
                    self.camera.pan(rel_x, rel_y)
            else:
                # Передаем rel_x и rel_y в функцию трансформации
                self._apply_motion(rel_x, rel_y)

        elif event.type == pygame.MOUSEBUTTONDOWN:
            self.mouse_pressed[event.button] = True
            # Зум: -1.0 приближает, 1.0 удаляет
            if event.button == 4: self.camera.zoom(-1.0)
            elif event.button == 5: self.camera.zoom(1.0)
            
            if event.button == 1 and not alt_pressed:
                if self.transform_mode:
                    self.transform_mode = None
                    self.transform_axis = None
                else:
                    self._handle_selection(event.pos)

        elif event.type == pygame.MOUSEBUTTONUP:
            self.mouse_pressed[event.button] = False

        elif event.type == pygame.KEYDOWN:
            ctrl_pressed = keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]

            # --- Создание объектов ---
            if event.key == pygame.K_1:
                self.scene.add_object(GeometryFactory.create_cube())
                
            elif event.key == pygame.K_2:
                self.scene.add_object(GeometryFactory.create_sphere())
                
            elif event.key == pygame.K_3:
                self.scene.add_object(GeometryFactory.create_pyramid())
                
            elif event.key == pygame.K_4:
                self.scene.add_object(GeometryFactory.create_plane())
                
            elif event.key == pygame.K_5:
                self.scene.add_object(GeometryFactory.create_cylinder())
               
            elif event.key == pygame.K_6:
                self.scene.add_object(GeometryFactory.create_cone())
                
            elif event.key == pygame.K_7:
                self.scene.add_object(GeometryFactory.create_torus())
                
            elif event.key == pygame.K_8:
                self.scene.add_object(GeometryFactory.create_capsule())
                

            # --- Режимы трансформации ---
            elif event.key == pygame.K_g: 
                self.transform_mode = 'G'; self.transform_axis = None
            elif event.key == pygame.K_r: 
                self.transform_mode = 'R'; self.transform_axis = None
            elif event.key == pygame.K_s: 
                self.transform_mode = 'S'; self.transform_axis = None
            elif event.key == pygame.K_ESCAPE: 
                self.transform_mode = None; self.transform_axis = None

            # --- Горячие клавиши (Undo и Export) ---
            
            elif event.key == pygame.K_e and ctrl_pressed:
                # ВАЖНО: убедись, что метод export_to_obj написан в этом же классе!
                self.scene.export_to_obj("my_scene.obj")
            # Выбор оси (X, Y, Z)
            if self.transform_mode:
                if event.key == pygame.K_x: self.transform_axis = 'X'
                elif event.key == pygame.K_y: self.transform_axis = 'Y'
                elif event.key == pygame.K_z: self.transform_axis = 'Z'

            # Переключение GLOBAL / LOCAL
            elif event.key == pygame.K_TAB:
                self.transform_space = 'LOCAL' if self.transform_space == 'GLOBAL' else 'GLOBAL'

            # Удаление
            elif event.key == pygame.K_x and not self.transform_mode:
                if self.scene.selected_object:
                    self.scene.objects.remove(self.scene.selected_object)
                    self.scene.selected_object = None

    def _apply_motion(self, rel_x, rel_y):
        obj = self.scene.selected_object
        if not obj: return
        
        # Коэффициент скорости
        f = 0.001 * self.camera.distance
        
        # 1. ПОЛУЧАЕМ ОСИ (Безопасный способ)
        if self.transform_space == 'GLOBAL':
            ax, ay, az = Vector3([1.0, 0.0, 0.0]), Vector3([0.0, 1.0, 0.0]), Vector3([0.0, 0.0, 1.0])
        else:
            # Берем оси прямо из матрицы вращения объекта
            m = Matrix44.from_quaternion(obj.rotation)
            # Это столбцы матрицы, отвечающие за локальные X, Y, Z
            ax = Vector3([m[0,0], m[0,1], m[0,2]])
            ay = Vector3([m[1,0], m[1,1], m[1,2]])
            az = Vector3([m[2,0], m[2,1], m[2,2]])

        # 2. ПЕРЕМЕЩЕНИЕ (G)
        if self.transform_mode == 'G':
            if self.transform_axis == 'X': obj.position += ax * (rel_x * f)
            elif self.transform_axis == 'Y': obj.position += ay * (-rel_y * f)
            elif self.transform_axis == 'Z': obj.position += az * (rel_y * f)
            else:
                # Свободное движение (относительно экрана)
                # Берем обратную матрицу вида, чтобы двигать в плоскости монитора
                v_inv = Matrix44.from_quaternion(Quaternion.from_matrix(self.camera.get_view_matrix())).inverse
                move_world = v_inv * Vector3([rel_x * f, -rel_y * f, 0.0])
                obj.position += move_world.xyz

        # 3. ВРАЩЕНИЕ (R) - ТУТ БЫЛА ОШИБКА
        elif self.transform_mode == 'R':
            angle = (rel_x + rel_y) * 0.02
            
            # Выбираем ось вращения
            if self.transform_axis == 'X': rot_axis = ax
            elif self.transform_axis == 'Y': rot_axis = ay
            elif self.transform_axis == 'Z': rot_axis = az
            else: rot_axis = Vector3([0.0, 1.0, 0.0]) # Глобальный верх

            # ГЛАВНАЯ ПРОВЕРКА: Если длина оси почти 0, не вращаем (защита от NaN)
            axis_len = np.linalg.norm(rot_axis)
            if axis_len > 0.0001:
                # Нормализуем ось только через numpy для стабильности
                norm_axis = rot_axis / axis_len
                # Создаем кватернион
                delta_rot = Quaternion.from_axis_rotation(norm_axis, angle)
                # Применяем вращение
                obj.rotation = delta_rot * obj.rotation

        # 4. МАСШТАБ (S)
        elif self.transform_mode == 'S':
            s_delta = 1.0 + rel_x * 0.01
            # Не даем масштабу стать нулевым
            if s_delta <= 0: s_delta = 0.001
            
            if self.transform_axis == 'X': obj.scale.x *= s_delta
            elif self.transform_axis == 'Y': obj.scale.y *= s_delta
            elif self.transform_axis == 'Z': obj.scale.z *= s_delta
            else: obj.scale *= s_delta

    def _handle_selection(self, pos):
        width, height = pygame.display.get_surface().get_size()
        
        # 1. Создаем временный буфер для "технического" рендера
        # Важно: используем текстуру без фильтрации (соседние пиксели не смешиваются)
        fbo_color = self.scene.ctx.texture((width, height), 4)
        fbo_depth = self.scene.ctx.depth_renderbuffer((width, height))
        fbo = self.scene.ctx.framebuffer(fbo_color, fbo_depth)
        
        fbo.use()
        self.scene.ctx.enable(moderngl.DEPTH_TEST)
        self.scene.ctx.disable(moderngl.BLEND) # Выключаем смешивание цветов!
        self.scene.ctx.clear(0, 0, 0, 1) # Фон черный (ID = 0)
        
        # Подготавливаем шейдер (только матрицы, без света)
        self.scene.program['view'].write(np.array(self.camera.get_view_matrix(), 'f4'))
        self.scene.program['projection'].write(np.array(self.camera.get_projection_matrix(), 'f4'))
        self.scene.program['selected'].value = False # Чтобы не подсвечивало при выборе

        # 2. Рисуем каждый объект своим "цифровым" цветом
        for i, obj in enumerate(self.scene.objects):
            obj_id = i + 1 # ID 0 зарезервирован для пустоты
            
            # Упаковываем ID в компоненты RGB (от 0 до 255)
            r = (obj_id & 0xFF) / 255.0
            g = ((obj_id >> 8) & 0xFF) / 255.0
            b = ((obj_id >> 16) & 0xFF) / 255.0
            
            self.scene.program['color'].value = (r, g, b)
            self.scene.program['model'].write(np.array(obj.get_model_matrix(), 'f4'))
            
            if obj.vao:
                obj.vao.render()

        # 3. Читаем пиксель строго под курсором
        # Инвертируем Y, так как в Pygame 0 вверху, а в OpenGL внизу
        pixel = fbo.read(viewport=(pos[0], height - pos[1], 1, 1), components=3, dtype='f1')
        
        fbo.release()
        self.scene.ctx.screen.use() # Возвращаемся на основной экран
        
        # 4. Расшифровываем ID обратно
        res = list(pixel)
        picked_id = res[0] + (res[1] << 8) + (res[2] << 16)
        
        if 0 < picked_id <= len(self.scene.objects):
            self.scene.selected_object = self.scene.objects[picked_id - 1]
        else:
            self.scene.selected_object = None
class Editor3D:
    def __init__(self, width=1280, height=720):
        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        
        self.screen = pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)
        pygame.display.set_caption("3D Editor")
        
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST)
        
        self.camera = Camera()
        self.camera.aspect = width / height
        
        self.scene = Scene(self.ctx)
        
        # ИСПРАВЛЕНО: Порядок аргументов теперь соответствует InputController(self, scene, camera)
        self.input_controller = InputController(self.scene, self.camera)
        
        # Создаем начальный куб
        cube = GeometryFactory.create_cube()
        self.scene.add_object(cube)
        
        self.clock = pygame.time.Clock()
        self.running = True
        self._print_help()

    

    def _print_help(self):
        print("""
        ╔═══════════════════════════════════════════════════════════════╗
        ║ УПРАВЛЕНИЕ:                                                   ║
        ║   Alt + ЛКМ         - Вращение камеры                         ║
        ║   Alt + ПКМ         - Панорамирование                         ║
        ║   Колесо мыши       - Зум (Ближе/Дальше)                       ║
        ║   TAB               - Переключение GLOBAL / LOCAL             ║
        ╠═══════════════════════════════════════════════════════════════╣
        ║ ОБЪЕКТЫ:                                                      ║
        ║   1 - Куб           2 - Сфера          3 - Пирамида           ║
        ║   4 - Плоскость     5 - Цилиндр        6 - Конус              ║
        ╠═══════════════════════════════════════════════════════════════╣
        ║ ТРАНСФОРМАЦИЯ:                                                ║
        ║   G / R / S         - Перемещение / Вращение / Масштаб        ║
        ║   X / Y / Z         - Ограничить по оси (в режиме трансф.)    ║
        ║   ЛКМ               - Применить изменения                     ║
        ║   Esc               - Отменить                                ║
        ║   X (без режима)    - Удалить выбранный объект                ║
        ╚═══════════════════════════════════════════════════════════════╝
        """)

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                else:
                    self.input_controller.handle_event(event)
            
            # Рендеринг
            self.ctx.clear(0.1, 0.1, 0.15)
            self.scene.render(self.camera)
            
            # Информация в заголовке
            fps = self.clock.get_fps()
            sel = self.scene.selected_object.name if self.scene.selected_object else "Нет"
            mode = self.input_controller.transform_mode or "Просмотр"
            space = self.input_controller.transform_space
            axis = self.input_controller.transform_axis or "Все"
            
            pygame.display.set_caption(
                f"3D Editor | {space} | Режим: {mode} (Ось: {axis}) | Выделено: {sel} | FPS: {fps:.0f}"
            )
            
            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()
    def _print_help(self):
        print("""
╔═══════════════════════════════════════════════════════════════╗
║           3D РЕДАКТОР - УПРАВЛЕНИЕ                            ║
╠═══════════════════════════════════════════════════════════════╣
║ КАМЕРА:                                                       ║
║   СКМ               - Вращение камеры                         ║
║   Shift + СКМ       - Панорамирование                         ║
║   Колесо            - Приближение/отдаление                   ║
╠═══════════════════════════════════════════════════════════════╣
║ ОБЪЕКТЫ:                                                      ║
║   1 - Куб           5 - Конус                                 ║
║   2 - Сфера         6 - Капсула                               ║
║   3 - Цилиндр       7 - Тор                                   ║
║   4 - Плоскость                                               ║
╠═══════════════════════════════════════════════════════════════╣
║ ТРАНСФОРМАЦИЯ:                                                ║
║   ЛКМ               - Выделить                                ║
║   G + движение      - Перемещение                             ║
║   R + движение      - Вращение                                ║
║   S + движение      - Масштабирование                         ║
║   X/Delete          - Удалить                                 ║
║   Esc               - Отменить                                 ║
╚═══════════════════════════════════════════════════════════════╝
        """)
    
    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                else:
                    self.input_controller.handle_event(event)
            self.ctx.clear(0.1, 0.1, 0.15)
            self.scene.render(self.camera)
            fps = self.clock.get_fps()
            sel = self.scene.selected_object.name if self.scene.selected_object else "Нет"
# Найди строку pygame.display.set_caption и замени на:
            space = self.input_controller.transform_space
            pygame.display.set_caption(f"3D Editor | {space} | FPS: {fps:.0f} | Выделено: {sel}")
            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()

if __name__ == "__main__":
    editor = Editor3D()
    editor.run()
