pip download bpy_helper==0.0.7 --dest ./renderformer_tools/wheels --no-deps
pip download trimesh==4.6.8 --dest ./renderformer_tools/wheels --no-deps
pip download networkx==3.4.2 --dest ./renderformer_tools/wheels --no-deps
# download for ["windows-x64", "macos-x64", "macos-arm64", "linux-x64"]
## h5py
pip download h5py==3.13.0 --dest ./renderformer_tools/wheels --only-binary=:all: --python-version=3.11 --platform=macosx_11_0_arm64 --no-deps
pip download h5py==3.13.0 --dest ./renderformer_tools/wheels --only-binary=:all: --python-version=3.11 --platform=macosx_10_9_x86_64 --no-deps
pip download h5py==3.13.0 --dest ./renderformer_tools/wheels --only-binary=:all: --python-version=3.11 --platform=win_amd64 --no-deps
pip download h5py==3.13.0 --dest ./renderformer_tools/wheels --only-binary=:all: --python-version=3.11 --platform=manylinux_2_17_x86_64 --no-deps

## pymeshlab
pip download pymeshlab==2023.12.post3 --dest ./renderformer_tools/wheels --only-binary=:all: --python-version=3.11 --platform=macosx_11_0_arm64 --no-deps
pip download pymeshlab==2023.12.post3 --dest ./renderformer_tools/wheels --only-binary=:all: --python-version=3.11 --platform=macosx_11_0_x86_64 --no-deps
pip download pymeshlab==2023.12.post3 --dest ./renderformer_tools/wheels --only-binary=:all: --python-version=3.11 --platform=win_amd64 --no-deps
pip download msvc_runtime==14.42.34433 --dest ./renderformer_tools/wheels --only-binary=:all: --python-version=3.11 --platform=win_amd64 --no-deps
pip download pymeshlab==2023.12.post3 --dest ./renderformer_tools/wheels --only-binary=:all: --python-version=3.11 --platform=manylinux_2_31_x86_64 --no-deps

## scipy
pip download scipy==1.15.2 --dest ./renderformer_tools/wheels --only-binary=:all: --python-version=3.11 --platform=macosx_14_0_arm64 --no-deps
pip download scipy==1.15.2 --dest ./renderformer_tools/wheels --only-binary=:all: --python-version=3.11 --platform=macosx_14_0_x86_64 --no-deps
pip download scipy==1.15.2 --dest ./renderformer_tools/wheels --only-binary=:all: --python-version=3.11 --platform=win_amd64 --no-deps
pip download scipy==1.15.2 --dest ./renderformer_tools/wheels --only-binary=:all: --python-version=3.11 --platform=manylinux_2_17_x86_64 --no-deps