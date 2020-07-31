# -*- mode: python ; coding: utf-8 -*-
import time
# block_cipher = pyi_crypto.PyiBlockCipher(key='')
block_cipher = None

added_files = [('resource/icon.ico', 'resource'), ('model.template', '.'), ('resource/VERSION', 'astor'), ('resource/VERSION', 'resource')]

a = Analysis(['app.py'],
             pathex=['.'],
             binaries=[],
             datas=added_files,
             hiddenimports=['numpy.core._dtype_ctypes', 'pkg_resources.py2_warn'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='app',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True,
          icon='resource/icon.ico')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='gpu-win64-{}'.format(time.strftime("%Y%m%d", time.localtime())))
