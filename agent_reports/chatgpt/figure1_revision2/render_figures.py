"""Export SVGs to PNG and vector PDF using installed librsvg/cairo (Linux)."""
import ctypes as c
from pathlib import Path
import xml.etree.ElementTree as ET

root = Path(__file__).parent
rsvg = c.CDLL('librsvg-2.so.2')
cairo = c.CDLL('libcairo.so.2')
gobject = c.CDLL('libgobject-2.0.so.0')

def sig(lib, name, args, result):
    f = getattr(lib, name)
    f.argtypes, f.restype = args, result
    return f

ptr = c.c_void_p
load = sig(rsvg, 'rsvg_handle_new_from_file', [c.c_char_p, ptr], ptr)
render = sig(rsvg, 'rsvg_handle_render_cairo', [ptr, ptr], c.c_int)
image = sig(cairo, 'cairo_image_surface_create', [c.c_int,c.c_int,c.c_int], ptr)
pdf = sig(cairo, 'cairo_pdf_surface_create', [c.c_char_p,c.c_double,c.c_double], ptr)
ctx = sig(cairo, 'cairo_create', [ptr], ptr)
png = sig(cairo, 'cairo_surface_write_to_png', [ptr,c.c_char_p], c.c_int)
destroy = sig(cairo, 'cairo_destroy', [ptr], None)
finish = sig(cairo, 'cairo_surface_finish', [ptr], None)
surface_destroy = sig(cairo, 'cairo_surface_destroy', [ptr], None)
unref = sig(gobject, 'g_object_unref', [ptr], None)
for path in sorted(root.glob('*.svg')):
    doc = ET.parse(path).getroot()
    w,h = int(doc.attrib['width']),int(doc.attrib['height'])
    handle = load(str(path).encode(), None)
    assert handle, path
    for ext in ['png','pdf']:
        out = path.with_suffix('.'+ext)
        surface = image(0,w,h) if ext=='png' else pdf(str(out).encode(),w,h)
        cr = ctx(surface)
        assert render(handle,cr),path
        if ext=='png': assert png(surface,str(out).encode())==0
        destroy(cr);finish(surface);surface_destroy(surface)
    unref(handle)
    print(path.stem)
