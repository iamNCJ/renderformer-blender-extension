bl_info = {
    "name": "Renderformer",
    "author": "NCJ",
    "version": (1, 0, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Renderformer",
    "description": "Export current Blender scene (single‑camera) to RenderFormer HDF5 — full material heuristics: emissive lights, vertex‑color, Principled BSDF constants, and custom Specular‑Roughness node‑group (no texture maps, no render)",
    "category": "Render",
}

# ------------------------------------------------------------
#  Std libs & deps
# ------------------------------------------------------------
import bpy
import bmesh
from mathutils import Color

import os
import math
from datetime import datetime, timezone

import numpy as np
import h5py
import pymeshlab
import trimesh

# ------------------------------------------------------------
#  Texture helpers (13‑channel per‑triangle tile)
#     [0‑2] diffuse, [3‑5] specular, [6] rough, [7‑9] normal, [10‑12] irradiance
# ------------------------------------------------------------
_TEXTURE_SIZE = 32
_x, _y = np.meshgrid(np.arange(_TEXTURE_SIZE), np.arange(_TEXTURE_SIZE), indexing='ij')
_TRI_MASK = (_x + _y) <= _TEXTURE_SIZE


def _make_tile(vec13):
    tile = np.zeros((13, _TEXTURE_SIZE, _TEXTURE_SIZE), dtype=np.float16)
    tile[:, _TRI_MASK] = vec13[:, None]
    return tile

# ------------------------------------------------------------
#  Shader‑graph inspection utilities
# ------------------------------------------------------------

def _principled_bsdf(mat):
    if not mat or not mat.use_nodes:
        return None
    # top‑level search
    for n in mat.node_tree.nodes:
        if n.type == 'BSDF_PRINCIPLED':
            return n
    # inside single‑level generic group "Group"
    grp = mat.node_tree.nodes.get('Group')
    if grp and hasattr(grp, 'node_tree'):
        for n in grp.node_tree.nodes:
            if n.type == 'BSDF_PRINCIPLED':
                return n
    return None


def _specular_rough_group(mat):
    if not mat or not mat.use_nodes:
        return None
    for n in mat.node_tree.nodes:
        if n.type == 'GROUP' and n.node_tree and n.node_tree.name.startswith('SpecularRoughnessBSDF'):  # to handle copied nodes like SpecularRoughnessBSDF.001
            return n
    return None


def _vertex_color_links(mat):
    if not mat or not mat.use_nodes:
        return False
    for n in mat.node_tree.nodes:
        if n.type == 'VERTEX_COLOR':
            for o in n.outputs:
                for l in o.links:
                    if l.to_socket.name == 'Base Color' or l.to_socket.name == 'Diffuse':
                        return True
    return False

# ------------------------------------------------------------
#  Constant vectors per material type
# ------------------------------------------------------------

def _vec_from_default(node):
    diff = [0.4, 0.4, 0.4]
    spec = [0.0, 0.0, 0.0]
    rough = 1.0
    normal = [0.5, 0.5, 1.0]
    emit = [0.0, 0.0, 0.0]
    return np.array(diff + spec + [rough] + normal + emit, dtype=np.float16)

def _vec_from_principled(bsdf):
    diff = list(bsdf.inputs['Base Color'].default_value[:3])
    if 'Specular IOR Level' in bsdf.inputs:  # Blender 4.0 key remap
        spec_scalar = float(bsdf.inputs['Specular IOR Level'].default_value)
    else:
        spec_scalar = float(bsdf.inputs['Specular'].default_value)
    spec = [spec_scalar] * 3
    rough = float(bsdf.inputs['Roughness'].default_value)
    emit = [0.0, 0.0, 0.0]
    normal = [0.5, 0.5, 1.0]
    return np.array(diff + spec + [rough] + normal + emit, dtype=np.float16)


def _vec_from_specularrough_node(node):
    diff = list(node.inputs['Diffuse'].default_value[:3])
    spec = list(node.inputs['Specular'].default_value[:3])
    rough = float(node.inputs['Roughness'].default_value)
    normal = [0.5, 0.5, 1.0]
    emit = [0.0, 0.0, 0.0]
    return np.array(diff + spec + [rough] + normal + emit, dtype=np.float16)


def _vec_for_emissive(color_rgb, strength):
    irr = (np.array(color_rgb) * strength).astype(np.float16)
    base = np.zeros(13, dtype=np.float16)
    base[0:3] = [1., 1., 1.]
    base[6] = 1.0
    base[7:10] = [0.5, 0.5, 1.0]
    base[10:13] = irr
    return base

# ------------------------------------------------------------
#  Vertex‑color helper (needs spec & rough constants)
# ------------------------------------------------------------

def _tri_texture_from_vertex_colors(mesh, diff_spec_rough):
    """
    mesh: bpy.types.Mesh
    diff_spec_rough: list[float]
    """
    color_layer = mesh.color_attributes.get("Color")
    # print(color_layer.data_type)
    if not color_layer:
        return None
    n_tri = len(mesh.polygons)
    spec = diff_spec_rough[3:6]
    rough = diff_spec_rough[6]
    tex = np.zeros((n_tri, 13, _TEXTURE_SIZE, _TEXTURE_SIZE), dtype=np.float16)
    for i, f in enumerate(mesh.polygons):
        col = np.zeros(3)
        for l in f.loop_indices:
            col_lin = np.array(color_layer.data[l].color[:3])
            col += np.array(Color(col_lin).from_scene_linear_to_srgb())
        col /= len(f.loop_indices)
        vec13 = np.concatenate([col, spec, [rough], [0.5,0.5,1.0], [0,0,0]]).astype(np.float16)
        tex[i] = _make_tile(vec13)
    return tex

# ------------------------------------------------------------
#  UI Panel
# ------------------------------------------------------------
class RENDERFORMER_PT_workspace_init(bpy.types.Panel):
    bl_label = 'RenderFormer Workspace Initialization'
    bl_idname = 'RENDERFORMER_PT_workspace_init'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Renderformer'

    def draw(self, ctx):
        layout = self.layout
        layout.operator('renderformer.initialize_all', icon='SEQ_SEQUENCER')
        layout.operator('renderformer.reset_scene', icon='FILE_REFRESH')
        layout.operator('renderformer.load_default_background_scene', icon='ADD')
        layout.operator('renderformer.load_default_light', icon='LIGHT')
        layout.operator('renderformer.load_default_camera', icon='CAMERA_DATA')

class RENDERFORMER_PT_normalization(bpy.types.Panel):
    bl_label = 'RenderFormer Normalization'
    bl_idname = 'RENDERFORMER_PT_normalization'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Renderformer'

    def draw(self, ctx):
        layout = self.layout
        layout.prop(ctx.scene, 'rf_target_scale', text='Target Scale')
        layout.operator('renderformer.normalize_scene', icon='ARROW_LEFTRIGHT')
        obj = ctx.object
        if obj:
            layout.label(text=f"Selected: {obj.name}")
            layout.operator('renderformer.normalize_selected_object', icon='OBJECT_DATA')
        else:
            layout.label(text="No object selected")

class RENDERFORMER_PT_export_single_frame(bpy.types.Panel):
    bl_label = 'Export Single Frame Export'
    bl_idname = 'RENDERFORMER_PT_export_single_frame'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Renderformer'

    def draw(self, ctx):
        layout = self.layout
        layout.prop(ctx.scene, 'rf_resolution')
        layout.prop(ctx.scene, 'rf_folder')
        layout.operator('renderformer.export_h5', icon='EXPORT')

class RENDERFORMER_PT_animation_export(bpy.types.Panel):
    bl_label = 'RenderFormer Animation Export'
    bl_idname = 'RENDERFORMER_PT_animation_export'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Renderformer'

    def draw(self, context):
        layout = self.layout
        layout.prop(context.scene, 'rf_anim_folder', text='Animation Folder')
        layout.prop(context.scene, 'rf_start_frame', text='Start Frame')
        layout.prop(context.scene, 'rf_end_frame', text='End Frame')
        layout.operator('renderformer.export_animation', icon='RENDER_ANIMATION')
        
        # Draw progress bar if export is in progress
        if context.window_manager.rf_export_progress > 0:
            layout.progress(factor=context.window_manager.rf_export_progress,
                          type='BAR',
                          text=f"{int(context.window_manager.rf_export_progress*100)}%")

class RENDERFORMER_PT_object_process(bpy.types.Panel):
    bl_label = 'RenderFormer Object Process'
    bl_idname = 'RENDERFORMER_PT_object_process'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Renderformer'

    def draw(self, context):
        layout = self.layout
        layout.prop(context.scene, 'rf_target_face_count', text="Target Face Count")
        layout.operator("renderformer.process_geometry", icon="MODIFIER")

class RENDERFORMER_PT_object_properties(bpy.types.Panel):
    bl_label = 'RenderFormer Object Properties'
    bl_idname = 'RENDERFORMER_PT_object_properties'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Renderformer'

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == 'MESH'

    def draw(self, context):
        layout = self.layout
        obj = context.object
        layout.label(text=f"Object: {obj.name}")
        layout.prop(obj, "rf_recompute_vn", text="Recompute Vertex Normal")
        layout.prop(obj, "rf_fix_object_normal", text="Force Fix Object Normal [!]")
        layout.prop(obj, "rf_use_smooth_shading", text="Use Smooth Shading [!]")

class RENDERFORMER_PT_scene_stats(bpy.types.Panel):
    bl_label = 'RenderFormer Scene Stats'
    bl_idname = 'RENDERFORMER_PT_scene_stats'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Renderformer'

    def draw(self, context):
        layout = self.layout
        total_tris = 0
        for obj in [o for o in bpy.data.objects if o.type == 'MESH' and not o.hide_render]:
            mesh = obj.to_mesh()
            # Count triangles, accounting for quads and n-gons
            for poly in mesh.polygons:
                # Number of triangles = number of vertices - 2
                total_tris += len(poly.vertices) - 2
            obj.to_mesh_clear()
        layout.label(text=f"Total Triangles: {total_tris:,}")

# ------------------------------------------------------------
#  Export operator
# ------------------------------------------------------------
class RENDERFORMER_OT_export(bpy.types.Operator):
    bl_idname = 'renderformer.export_h5'
    bl_label = 'Export Scene to HDF5'
    bl_description = 'Export the current Blender scene to a RenderFormer HDF5 file.'
    bl_options = {'REGISTER'}

    frame_override: bpy.props.IntProperty(
        name="Frame Override",
        description="If set, use this frame number (6 digits) for file naming",
        default=-1
    )

    def execute(self, ctx):
        folder = bpy.path.abspath(ctx.scene.rf_folder or '//renderformer')
        os.makedirs(folder, exist_ok=True)

        tris_all, vn_all, tex_all = [], [], []

        for obj in [o for o in bpy.data.objects if o.type == 'MESH']:
            # Skip objects that are hidden from render
            if obj.hide_render:
                print(f"{obj.name} is hidden from render")
                continue
                
            print(obj.name)
            depsgraph = bpy.context.evaluated_depsgraph_get()
            eval_obj = obj.evaluated_get(depsgraph)

            mat = eval_obj.material_slots[0].material if eval_obj.material_slots else None
            bsdf = _principled_bsdf(mat)
            sr_node = _specular_rough_group(mat)

            # ---- classify material -----------------------------------
            is_light = False
            if bsdf:
                # Blender 4: Emission Color/Strength; Blender 3.x: Emission / Emission Strength
                if ('Emission Strength' in bsdf.inputs and bsdf.inputs['Emission Strength'].default_value > 0.0):
                    is_light = True
                    emis_strength = float(bsdf.inputs['Emission Strength'].default_value)
                    emis_col = bsdf.inputs.get('Emission Color', bsdf.inputs.get('Emission')).default_value[:3]
            is_vcol = _vertex_color_links(mat)

            # ---- triangulate geometry -------------------------------
            bm = bmesh.new()
            bm.from_mesh(eval_obj.to_mesh())
            bmesh.ops.triangulate(bm, faces=bm.faces[:])
            bm.transform(eval_obj.matrix_world)
            bm.verts.ensure_lookup_table()
            bm.faces.ensure_lookup_table()

            n_tri = len(bm.faces)
            tri_xyz = np.zeros((n_tri, 3, 3), dtype=np.float32)
            tri_vn  = np.zeros_like(tri_xyz)
            for i, f in enumerate(bm.faces):
                for j, v in enumerate(f.verts):
                    tri_xyz[i, j] = v.co[:]
                    tri_vn[i, j]  = v.normal[:]

            # ---- world normal transform ----
            normal_matrix = eval_obj.matrix_world.inverted().transposed().to_3x3()
            tri_vn = tri_vn @ np.array(normal_matrix, dtype=np.float32).T

            # Force fix object normal if enabled
            if obj.rf_fix_object_normal and not is_light:
                is_vcol = False  # mesh topology has been changed, vcol is not valid anymore

                # Create trimesh object with full processing
                tri_mesh = trimesh.Trimesh(vertices=tri_xyz.reshape(-1, 3), 
                                         faces=np.arange(len(tri_xyz) * 3).reshape(-1, 3),
                                         process=True)
                # Force recompute face normals
                tri_mesh.fix_normals()

                # Get the recomputed vertices and normals
                triangles = tri_mesh.triangles
                vn = tri_mesh.vertex_normals[tri_mesh.faces]
                tri_xyz = triangles.reshape(-1, 3, 3)
                tri_vn = vn

            # Recompute vertex normals if enabled
            if obj.rf_recompute_vn:
                # Create trimesh object
                if obj.rf_use_smooth_shading:
                    tri_mesh = trimesh.Trimesh(
                        vertices=tri_xyz.reshape(-1, 3),
                        faces=np.arange(len(tri_xyz) * 3).reshape(-1, 3),
                        process=True
                    )
                    tri_mesh = trimesh.graph.smooth_shade(tri_mesh, angle=np.radians(30))
                    is_vcol = False
                else:
                    tri_mesh = trimesh.Trimesh(
                        vertices=tri_xyz.reshape(-1, 3),
                        faces=np.arange(len(tri_xyz) * 3).reshape(-1, 3),
                        process=False
                    )
                tri_xyz = tri_mesh.vertices[tri_mesh.faces]
                tri_vn = tri_mesh.vertex_normals[tri_mesh.faces]

            tris_all.append(tri_xyz)
            vn_all.append(tri_vn)

            # ---- build texture --------------------------------------
            if is_light:
                vec13 = _vec_for_emissive(emis_col, emis_strength)
                tex_all.append(np.repeat(_make_tile(vec13)[None, ...], n_tri, axis=0))
            elif is_vcol:
                const_vec = _vec_from_specularrough_node(sr_node)
                mesh = obj.to_mesh()
                tex = _tri_texture_from_vertex_colors(mesh, const_vec)
                tex_all.append(tex)
            elif sr_node is not None:  # custom SpecularRoughnessBSDF
                const_vec = _vec_from_specularrough_node(sr_node)
                tex_all.append(np.repeat(_make_tile(const_vec)[None, ...], n_tri, axis=0))
            elif bsdf is not None:
                const_vec = _vec_from_principled(bsdf)
                tex_all.append(np.repeat(_make_tile(const_vec)[None, ...], n_tri, axis=0))
            else:
                # unknown shader setup — fallback gray color blank bsdf
                const_vec = _vec_from_default(bsdf)
                tex_all.append(np.repeat(_make_tile(const_vec)[None, ...], n_tri, axis=0))
            bm.free()
            eval_obj.to_mesh_clear()

        if not tris_all:
            self.report({'ERROR'}, 'No mesh geometry found')
            return {'CANCELLED'}

        triangles = np.concatenate(tris_all, axis=0)
        vns       = np.concatenate(vn_all,  axis=0)
        textures  = np.concatenate(tex_all, axis=0)

        cam = ctx.scene.camera or next((o for o in bpy.data.objects if o.type == 'CAMERA'), None)
        if not cam:
            self.report({'ERROR'}, 'No camera in scene')
            return {'CANCELLED'}
        c2w = np.array(cam.matrix_world, dtype=np.float32)
        fov = math.degrees(cam.data.angle)

        # File naming logic
        if self.frame_override >= 0:
            frame_str = f"{self.frame_override:06d}"
            out_path = os.path.join(folder, f'{frame_str}.h5')
        else:
            idx = max([int(os.path.splitext(f)[0]) for f in os.listdir(folder) if f.split('.')[0].isdigit()] or [-1]) + 1
            out_path = os.path.join(folder, f'{idx}.h5')
        with h5py.File(out_path, 'w') as h5:
            h5.create_dataset('triangles', data=triangles, compression='gzip', compression_opts=9)
            h5.create_dataset('vn', data=vns, compression='gzip', compression_opts=9)
            h5.create_dataset('texture', data=textures, compression='gzip', compression_opts=9)
            h5.create_dataset('c2w', data=c2w[None], compression='gzip', compression_opts=9)
            h5.create_dataset('fov', data=np.array([fov], dtype=np.float32), compression='gzip', compression_opts=9)
            h5.attrs['created'] = datetime.now(timezone.utc).isoformat()
            h5.attrs['renderformer'] = '0.8.0'
        print(f'Exported to {out_path}, triangles: {triangles.shape}, vns: {vns.shape}, textures: {textures.shape}')
        self.report({'INFO'}, f'Exported to {out_path}')
        return {'FINISHED'}

# ------------------------------------------------------------
#  Reset scene operator
# ------------------------------------------------------------
class RENDERFORMER_OT_reset_scene(bpy.types.Operator):
    bl_idname = 'renderformer.reset_scene'
    bl_label = 'Reset Scene'
    bl_description = 'Reset the scene to its initial state.'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        # remove default objects
        for obj in bpy.data.objects:
            bpy.data.objects.remove(obj, do_unlink=True)

        # enable face orientation overlay
        bpy.context.space_data.overlay.show_face_orientation = True

        resolution = ctx.scene.rf_resolution
        bpy.context.scene.render.resolution_x = resolution
        bpy.context.scene.render.resolution_y = resolution
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.samples = 128
        bpy.context.scene.render.film_transparent = True
        bpy.context.scene.render.image_settings.color_mode = 'RGBA'
        bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'

        bpy.context.scene.world.node_tree.nodes["Background"].inputs[1].default_value = 0.  # remove all ambient

        # import material
        from bpy_helper.material import create_specular_roughness_bsdf
        create_specular_roughness_bsdf()

        self.report({'INFO'}, "Scene has been reset!")
        return {'FINISHED'}

# ------------------------------------------------------------
#  Load default scene operator
# ------------------------------------------------------------
class RENDERFORMER_OT_load_default_background_scene(bpy.types.Operator):
    bl_idname = 'renderformer.load_default_background_scene'
    bl_label = 'Load Default Background Scene'
    bl_description = 'Load the default scene with basic meshes.'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        # Get the addon directory
        addon_dir = os.path.dirname(os.path.realpath(__file__))
        data_dir = os.path.join(addon_dir, 'data', 'meshes')

        # Define the mesh paths and their corresponding names
        mesh_paths = {
            # 'ceiling': os.path.join(data_dir, 'ceiling.obj'),
            'plane': os.path.join(data_dir, 'plane.obj'),
            'wall0': os.path.join(data_dir, 'wall0.obj'),
            'wall1': os.path.join(data_dir, 'wall1.obj'),
            'wall2': os.path.join(data_dir, 'wall2.obj')
        }

        # Import each mesh and set up its material
        for obj_name, mesh_path in mesh_paths.items():
            if not os.path.exists(mesh_path):
                self.report({'ERROR'}, f"Mesh file not found: {mesh_path}")
                continue
                
            # Import the mesh
            from bpy_helper.scene import import_3d_model
            imported_obj = import_3d_model(mesh_path)
            
            # Get the imported object
            imported_obj = bpy.context.selected_objects[0]
            imported_obj.name = obj_name
            
            # Clear existing materials and add new one
            imported_obj.data.materials.clear()
            from bpy_helper.material import create_specular_roughness_material
            material = create_specular_roughness_material(
                diffuse_color=(0.4, 0.4, 0.4),
            )
            imported_obj.data.materials.append(material)
            
            # Set rotation to zero
            imported_obj.rotation_mode = 'XYZ'
            imported_obj.rotation_euler = (0.0, 0.0, 0.0)
            imported_obj.scale = (0.5, 0.5, 0.5)

        self.report({'INFO'}, "Default scene loaded!")
        return {'FINISHED'}

# ------------------------------------------------------------
#  Load default light operator
# ------------------------------------------------------------
class RENDERFORMER_OT_load_default_light(bpy.types.Operator):
    bl_idname = 'renderformer.load_default_light'
    bl_label = 'Add Default Light'
    bl_description = 'Add a default ceiling light.'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        # Get the addon directory
        addon_dir = os.path.dirname(os.path.realpath(__file__))
        light_mesh_path = os.path.join(addon_dir, 'data', 'meshes', 'tri.obj')

        # Import the light mesh
        from bpy_helper.scene import import_3d_model
        imported_obj = import_3d_model(light_mesh_path)
        
        # Get the imported object
        light_obj = bpy.context.selected_objects[0]
        light_obj.name = 'light_0'
        
        # Set transform
        light_obj.location = (0.0, 0.0, 2.1)
        light_obj.rotation_euler = (0.0, 0.0, 0.0)
        light_obj.scale = (2.5, 2.5, 2.5)
        
        # Clear existing materials and add emissive material
        light_obj.data.materials.clear()
        from bpy_helper.material import create_white_emmissive_material
        material = create_white_emmissive_material(strength=4500.0)
        light_obj.data.materials.append(material)

        self.report({'INFO'}, "Default light added!")
        return {'FINISHED'}

# ------------------------------------------------------------
#  Load default camera operator
# ------------------------------------------------------------
class RENDERFORMER_OT_load_default_camera(bpy.types.Operator):
    bl_idname = 'renderformer.load_default_camera'
    bl_label = 'Add Default Camera'
    bl_description = 'Add a default camera with predefined position and settings.'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        from bpy_helper.camera import create_camera, look_at_to_c2w
        
        # Default camera configuration
        camera_pos = np.array([0.0, -2.0, 0.0])
        look_at = np.array([0.0, 0.0, 0.0])
        up = np.array([0.0, 0.0, 1.0])
        fov = 37.5
        
        # Create camera transform and camera object
        c2w = look_at_to_c2w(camera_pos, look_at, up)
        camera = create_camera(c2w, fov)
        bpy.context.scene.camera = camera

        self.report({'INFO'}, "Default camera added!")
        return {'FINISHED'}

# ------------------------------------------------------------
#  Normalize scene operator
# ------------------------------------------------------------
class RENDERFORMER_OT_normalize_scene(bpy.types.Operator):
    bl_idname = 'renderformer.normalize_scene'
    bl_label = 'Normalize Scene'
    bl_description = 'Normalize all objects to fit in a -0.5 to 0.5 cube.'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        from bpy_helper.scene import normalize_scene
        normalize_scene(target_scale=ctx.scene.rf_target_scale)
        self.report({'INFO'}, "Scene normalized!")
        return {'FINISHED'}

class RENDERFORMER_OT_process_geometry(bpy.types.Operator):
    bl_idname = "renderformer.process_geometry"
    bl_label = "Process Geometry"
    bl_description = "Use PyMeshLab to process selected mesh."
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.object
        mesh = obj.to_mesh()
        # Check if the mesh is already triangulated
        for p in mesh.polygons:
            if len(p.vertices) != 3:
                self.report({'ERROR'}, "Mesh is not triangulated")
                return {'CANCELLED'}
        verts = np.array([v.co for v in mesh.vertices])
        faces = np.array([p.vertices for p in mesh.polygons])
        tri = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(tri.vertices, tri.faces))

        count = context.scene.rf_target_face_count
        ms.meshing_isotropic_explicit_remeshing(
            targetlen=pymeshlab.PercentageValue(0.5),
            featuredeg=30,
            adaptive=False
        )
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=count,
            qualitythr=1.0
        )

        m_out = ms.current_mesh()
        v_out = m_out.vertex_matrix()
        f_out = m_out.face_matrix()

        new_mesh = bpy.data.meshes.new(name="processed_mesh")
        new_mesh.from_pydata(v_out.tolist(), [], f_out.tolist())
        new_mesh.update()
        obj.data = new_mesh
        self.report({'INFO'}, f"Remeshed: {len(v_out)} verts, {len(f_out)} faces")
        return {'FINISHED'}

class RENDERFORMER_OT_export_animation(bpy.types.Operator):
    bl_idname = 'renderformer.export_animation'
    bl_label = 'Export Animation to HDF5'
    bl_description = 'Export a range of frames as HDF5 files with a progress bar.'
    bl_options = {'REGISTER'}
    _timer = None

    def execute(self, ctx):
        start_frame = ctx.scene.rf_start_frame
        end_frame = ctx.scene.rf_end_frame
        
        if start_frame > end_frame:
            self.report({'ERROR'}, 'Start frame must be less than or equal to end frame')
            return {'CANCELLED'}
            
        # Save original folder and set animation folder
        self.original_folder = ctx.scene.rf_folder
        ctx.scene.rf_folder = ctx.scene.rf_anim_folder
        
        # Initialize progress
        ctx.window_manager.rf_export_progress = 0.0
        self.total_frames = end_frame - start_frame + 1
        self.current_frame = start_frame
        
        # Add modal timer
        wm = ctx.window_manager
        wm.modal_handler_add(self)
        self._timer = wm.event_timer_add(0.1, window=ctx.window)
        
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'TIMER':
            if self.current_frame <= context.scene.rf_end_frame:
                # Export current frame with frame_override
                context.scene.frame_set(self.current_frame)
                bpy.ops.renderformer.export_h5(frame_override=self.current_frame)
                
                # Update progress
                progress = (self.current_frame - context.scene.rf_start_frame) / self.total_frames
                context.window_manager.rf_export_progress = progress
                
                self.current_frame += 1
                return {'RUNNING_MODAL'}
            else:
                # Export complete
                context.window_manager.event_timer_remove(self._timer)
                context.scene.rf_folder = self.original_folder
                context.window_manager.rf_export_progress = 0.0
                self.report({'INFO'}, f'Exported frames {context.scene.rf_start_frame} to {context.scene.rf_end_frame}')
                return {'FINISHED'}
        
        return {'PASS_THROUGH'}

    def cancel(self, context):
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
        if hasattr(self, 'original_folder'):
            context.scene.rf_folder = self.original_folder
        context.window_manager.rf_export_progress = 0.0

class RENDERFORMER_OT_normalize_selected_object(bpy.types.Operator):
    bl_idname = 'renderformer.normalize_selected_object'
    bl_label = 'Normalize Selected Object'
    bl_description = 'Normalize only the selected object to fit in a -0.5 to 0.5 cube.'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        obj = ctx.object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, 'No mesh object selected')
            return {'CANCELLED'}
        from bpy_helper.scene import scene_bbox
        bbox_min, bbox_max = scene_bbox(obj)
        scale = ctx.scene.rf_target_scale
        center = (0, 0, 0)
        size = max(bbox_max - bbox_min)
        bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME', center='MEDIAN')
        obj.location = center
        obj.scale = (scale / size * obj.scale[0], scale / size * obj.scale[1], scale / size * obj.scale[2])
        self.report({'INFO'}, f"Object '{obj.name}' normalized!")
        return {'FINISHED'}

class RENDERFORMER_OT_initialize_all(bpy.types.Operator):
    bl_idname = 'renderformer.initialize_all'
    bl_label = 'Initialize All'
    bl_description = 'Run Reset Scene, Load Default Background Scene, Add Default Light, and Add Default Camera in sequence.'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        bpy.ops.renderformer.reset_scene()
        bpy.ops.renderformer.load_default_background_scene()
        bpy.ops.renderformer.load_default_light()
        bpy.ops.renderformer.load_default_camera()
        self.report({'INFO'}, 'All initialization steps completed!')
        return {'FINISHED'}

# ------------------------------------------------------------
#  Registration
# ------------------------------------------------------------
classes = (
    RENDERFORMER_PT_workspace_init,
    RENDERFORMER_PT_normalization,
    RENDERFORMER_PT_export_single_frame,
    RENDERFORMER_PT_animation_export,
    RENDERFORMER_PT_object_process,
    RENDERFORMER_PT_object_properties,
    RENDERFORMER_PT_scene_stats,
    RENDERFORMER_OT_export,
    RENDERFORMER_OT_export_animation,
    RENDERFORMER_OT_reset_scene,
    RENDERFORMER_OT_load_default_background_scene,
    RENDERFORMER_OT_load_default_light,
    RENDERFORMER_OT_load_default_camera,
    RENDERFORMER_OT_normalize_scene,
    RENDERFORMER_OT_normalize_selected_object,
    RENDERFORMER_OT_process_geometry,
    RENDERFORMER_OT_initialize_all,
)

def register():
    bpy.types.Scene.rf_resolution = bpy.props.IntProperty(name='Resolution', default=512, min=64, max=8192)
    bpy.types.Scene.rf_folder = bpy.props.StringProperty(name='Output Folder', subtype='DIR_PATH', default='//')
    bpy.types.Scene.rf_anim_folder = bpy.props.StringProperty(name='Animation Folder', subtype='DIR_PATH', default='//renderformer_anim')
    bpy.types.Scene.rf_target_scale = bpy.props.FloatProperty(name='Target Scale', default=1.0, min=0.1, max=10.0)
    bpy.types.Scene.rf_target_face_count = bpy.props.IntProperty(name='Target Face Count', default=2048, min=100, max=100000)
    bpy.types.Scene.rf_start_frame = bpy.props.IntProperty(name='Start Frame', default=1, min=1)
    bpy.types.Scene.rf_end_frame = bpy.props.IntProperty(name='End Frame', default=250, min=1)
    bpy.types.WindowManager.rf_export_progress = bpy.props.FloatProperty(default=0.0)
    bpy.types.Object.rf_recompute_vn = bpy.props.BoolProperty(name='Recompute Vertex Normal', default=True)
    bpy.types.Object.rf_fix_object_normal = bpy.props.BoolProperty(
        name='Fix Object Normal',
        description='Using trimesh, will break mesh topology and force homogeneous shading',
        default=False
    )
    bpy.types.Object.rf_use_smooth_shading = bpy.props.BoolProperty(
        name='Use Smooth Shading',
        description='Enable smooth shading for this object (will break mesh topology and lost per-face color)',
        default=True
    )
    for c in classes:
        bpy.utils.register_class(c)

def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
    del bpy.types.Scene.rf_resolution, bpy.types.Scene.rf_folder, bpy.types.Scene.rf_target_scale
    del bpy.types.Scene.rf_target_face_count
    del bpy.types.Scene.rf_start_frame, bpy.types.Scene.rf_end_frame, bpy.types.Scene.rf_anim_folder
    del bpy.types.WindowManager.rf_export_progress
    del bpy.types.Object.rf_recompute_vn
    del bpy.types.Object.rf_fix_object_normal
    del bpy.types.Object.rf_use_smooth_shading

if __name__ == '__main__':
    register()
