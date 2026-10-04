"""Create RTX SYLPH desktop launcher and place it left of the astronaut."""
import ctypes
import os
import time
from ctypes import POINTER, Structure, byref, c_char, c_long, c_void_p, c_wchar, sizeof, windll

import pythoncom
import win32api
import win32con
import win32gui
import win32process
from PIL import Image
from win32com.shell import shell, shellcon

DESKTOP = os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop")
if not os.path.isdir(DESKTOP):
    DESKTOP = os.path.join(os.path.expanduser("~"), "Desktop")

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "assets", "SYLPH_Icon.png")
ICO = os.path.join(HERE, "assets", "SYLPH_launcher.ico")
BAT = os.path.join(HERE, "run_sylph.bat")
WORK = HERE
LNK = os.path.join(DESKTOP, "RTX SYLPH.lnk")
# 1920x1080 grid (76x87). Astronaut stands at the far-right rock;
# 1768,524 is one column left of him, at torso height, empty slot.
TARGET_X, TARGET_Y = 1768, 524

LVM_FIRST = 0x1000
LVM_GETITEMCOUNT = LVM_FIRST + 4
LVM_GETITEMTEXTW = LVM_FIRST + 115
LVM_SETITEMPOSITION = LVM_FIRST + 15
LVIF_TEXT = 0x0001

PROCESS_VM_OPERATION = 0x0008
PROCESS_VM_READ = 0x0010
PROCESS_VM_WRITE = 0x0020
PROCESS_QUERY_INFORMATION = 0x0400
MEM_COMMIT = 0x1000
MEM_RELEASE = 0x8000
PAGE_READWRITE = 0x04


class LVITEMW(Structure):
    _fields_ = [
        ("mask", ctypes.c_uint),
        ("iItem", ctypes.c_int),
        ("iSubItem", ctypes.c_int),
        ("state", ctypes.c_uint),
        ("stateMask", ctypes.c_uint),
        ("pszText", c_void_p),
        ("cchTextMax", ctypes.c_int),
        ("iImage", ctypes.c_int),
        ("lParam", ctypes.c_void_p),
        ("iIndent", ctypes.c_int),
        ("iGroupId", ctypes.c_int),
        ("cColumns", ctypes.c_uint),
        ("puColumns", c_void_p),
        ("piColFmt", c_void_p),
        ("iGroup", ctypes.c_int),
    ]


kernel32 = windll.kernel32


def make_ico():
    img = Image.open(SRC).convert("RGBA")
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = max(0, int(h * 0.04))
    if top + side > h:
        top = h - side
    crop = img.crop((left, top, left + side, top + side))
    crop.save(
        ICO,
        format="ICO",
        sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )
    print("ico", ICO, os.path.getsize(ICO))


def make_shortcut():
    pythoncom.CoInitialize()
    shortcut = pythoncom.CoCreateInstance(
        shell.CLSID_ShellLink, None, pythoncom.CLSCTX_INPROC_SERVER, shell.IID_IShellLink
    )
    shortcut.SetPath(BAT)
    shortcut.SetWorkingDirectory(WORK)
    shortcut.SetDescription("Launch RTX SYLPH — double-click")
    shortcut.SetIconLocation(ICO, 0)
    persist = shortcut.QueryInterface(pythoncom.IID_IPersistFile)
    persist.Save(LNK, 0)
    shell.SHChangeNotify(shellcon.SHCNE_ASSOCCHANGED, shellcon.SHCNF_IDLIST, None, None)
    print("lnk", LNK)


def find_listview():
    progman = win32gui.FindWindow("Progman", None)
    defview = win32gui.FindWindowEx(progman, 0, "SHELLDLL_DefView", None)
    worker = 0
    while not defview:
        worker = win32gui.FindWindowEx(0, worker, "WorkerW", None)
        if not worker:
            break
        defview = win32gui.FindWindowEx(worker, 0, "SHELLDLL_DefView", None)
    if not defview:
        return None
    return win32gui.FindWindowEx(defview, 0, "SysListView32", None)


def listview_item_text(lv, index, max_chars=512):
    tid, pid = win32process.GetWindowThreadProcessId(lv)
    access = PROCESS_VM_OPERATION | PROCESS_VM_READ | PROCESS_VM_WRITE | PROCESS_QUERY_INFORMATION
    hproc = kernel32.OpenProcess(access, False, pid)
    if not hproc:
        raise OSError("OpenProcess explorer failed")
    try:
        buf_size = max_chars * 2
        remote_item = kernel32.VirtualAllocEx(hproc, 0, sizeof(LVITEMW), MEM_COMMIT, PAGE_READWRITE)
        remote_text = kernel32.VirtualAllocEx(hproc, 0, buf_size, MEM_COMMIT, PAGE_READWRITE)
        if not remote_item or not remote_text:
            raise OSError("VirtualAllocEx failed")
        item = LVITEMW()
        item.mask = LVIF_TEXT
        item.iItem = index
        item.iSubItem = 0
        item.pszText = remote_text
        item.cchTextMax = max_chars
        written = ctypes.c_size_t()
        kernel32.WriteProcessMemory(hproc, remote_item, byref(item), sizeof(item), byref(written))
        win32gui.SendMessage(lv, LVM_GETITEMTEXTW, index, remote_item)
        raw = ctypes.create_unicode_buffer(max_chars)
        kernel32.ReadProcessMemory(hproc, remote_text, raw, buf_size, byref(written))
        return raw.value
    finally:
        if remote_item:
            kernel32.VirtualAllocEx  # noqa
            kernel32.VirtualFreeEx(hproc, remote_item, 0, MEM_RELEASE)
        if remote_text:
            kernel32.VirtualFreeEx(hproc, remote_text, 0, MEM_RELEASE)
        kernel32.CloseHandle(hproc)


def place_icon():
    lv = find_listview()
    if not lv:
        print("no SysListView32")
        return False
    count = win32gui.SendMessage(lv, LVM_GETITEMCOUNT, 0, 0)
    print("icon_count", count)
    found = None
    for i in range(count):
        try:
            name = listview_item_text(lv, i)
        except Exception as e:
            print("read fail", i, e)
            continue
        key = (name or "").strip().lower()
        if "sylph" in key:
            print("candidate", i, repr(name))
        if key in ("rtx sylph", "rtx sylph.lnk"):
            found = i
    if found is None:
        print("RTX SYLPH not in listview names")
        return False
    lparam = win32api.MAKELONG(TARGET_X, TARGET_Y)
    win32gui.SendMessage(lv, LVM_SETITEMPOSITION, found, lparam)
    print("moved", found, "to", TARGET_X, TARGET_Y)
    return True


if __name__ == "__main__":
    make_ico()
    make_shortcut()
    time.sleep(0.8)
    try:
        ok = place_icon()
        print("placed" if ok else "not_placed")
    except Exception as e:
        print("place failed", type(e).__name__, e)
