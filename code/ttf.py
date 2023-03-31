from importlib import reload
import os
import numpy as np
import bezier
import freetype as ft
import pydiffvg
import torch
import save_svg

device = torch.device("cuda" if (
        torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")

reload(bezier)

def fix_single_svg(svg_path, all_word=False):
    target_h_letter = 360
    target_canvas_width, target_canvas_height = 600, 600

    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(svg_path)

    letter_h = canvas_height
    letter_w = canvas_width

    if all_word:
        if letter_w > letter_h:
            scale_canvas_w = target_h_letter / letter_w
            hsize = int(letter_h * scale_canvas_w)
            scale_canvas_h = hsize / letter_h
        else:
            scale_canvas_h = target_h_letter / letter_h
            wsize = int(letter_w * scale_canvas_h)
            scale_canvas_w = wsize / letter_w
    else:
        scale_canvas_h = target_h_letter / letter_h
        wsize = int(letter_w * scale_canvas_h)
        scale_canvas_w = wsize / letter_w

    for num, p in enumerate(shapes):
        p.points[:, 0] = p.points[:, 0] * scale_canvas_w
        p.points[:, 1] = p.points[:, 1] * scale_canvas_h + target_h_letter

    w_min, w_max = min([torch.min(p.points[:, 0]) for p in shapes]), max([torch.max(p.points[:, 0]) for p in shapes])
    h_min, h_max = min([torch.min(p.points[:, 1]) for p in shapes]), max([torch.max(p.points[:, 1]) for p in shapes])

    for num, p in enumerate(shapes):
        p.points[:, 0] = p.points[:, 0] + target_canvas_width/2 - int(w_min + (w_max - w_min) / 2)
        p.points[:, 1] = p.points[:, 1] + target_canvas_height/2 - int(h_min + (h_max - h_min) / 2)

    output_path = f"{svg_path[:-4]}_scaled.svg"
    save_svg.save_svg(output_path, target_canvas_width, target_canvas_height, shapes, shape_groups)


def normalize_letter_size(dest_path, font, txt):
    fontname = os.path.splitext(os.path.basename(font))[0]
    for i, c in enumerate(txt):
        fname = f"{dest_path}/{fontname}_{c}.svg"
        fname = fname.replace(" ", "_")
        fix_single_svg(fname)

    fname = f"{dest_path}/{fontname}_{txt}.svg"
    fname = fname.replace(" ", "_")
    fix_single_svg(fname, all_word=True)


def glyph_to_cubics(face, x=0):
    ''' Convert current font face glyph to cubic beziers'''

    def linear_to_cubic(Q):
        a, b = Q
        return [a + (b - a) * t for t in np.linspace(0, 1, 4)]

    def quadratic_to_cubic(Q):
        return [Q[0],
                Q[0] + (2 / 3) * (Q[1] - Q[0]),
                Q[2] + (2 / 3) * (Q[1] - Q[2]),
                Q[2]]

    beziers = []
    pt = lambda p: np.array([p.x + x, -p.y])  # Flipping here since freetype has y-up
    last = lambda: beziers[-1][-1]

    def move_to(a, beziers):
        beziers.append([pt(a)])

    def line_to(a, beziers):
        Q = linear_to_cubic([last(), pt(a)])
        beziers[-1] += Q[1:]

    def conic_to(a, b, beziers):
        Q = quadratic_to_cubic([last(), pt(a), pt(b)])
        beziers[-1] += Q[1:]

    def cubic_to(a, b, c, beziers):
        beziers[-1] += [pt(a), pt(b), pt(c)]

    face.glyph.outline.decompose(beziers, move_to=move_to, line_to=line_to, conic_to=conic_to, cubic_to=cubic_to)
    beziers = [np.array(C).astype(float) for C in beziers]
    return beziers


def font_string_to_beziers(font, txt, size=30, spacing=1.0, merge=True, target_control=None):
    ''' Load a font and convert the outlines for a given string to cubic bezier curves,
        if merge is True, simply return a list of all bezier curves,
        otherwise return a list of lists with the bezier curves for each glyph'''

    face = ft.Face(font)
    face.set_char_size(64 * size)
    slot = face.glyph

    x = 0
    beziers = []
    previous = 0
    for c in txt:
        face.load_char(c, ft.FT_LOAD_DEFAULT | ft.FT_LOAD_NO_BITMAP)
        bez = glyph_to_cubics(face, x)

        # Check number of control points if desired
        if target_control is not None:
            if c in target_control.keys():
                nctrl = np.sum([len(C) for C in bez])
                while nctrl < target_control[c]:
                    longest = np.max(
                        sum([[bezier.approx_arc_length(b) for b in bezier.chain_to_beziers(C)] for C in bez], []))
                    thresh = longest * 0.5
                    bez = [bezier.subdivide_bezier_chain(C, thresh) for C in bez]
                    nctrl = np.sum([len(C) for C in bez])
                    print(nctrl)

        if merge:
            beziers += bez
        else:
            beziers.append(bez)

        kerning = face.get_kerning(previous, c)
        x += (slot.advance.x + kerning.x) * spacing
        previous = c

    return beziers


def bezier_chain_to_commands(C, closed=True):
    curves = bezier.chain_to_beziers(C)
    cmds = 'M %f %f ' % (C[0][0], C[0][1])
    n = len(curves)
    for i, bez in enumerate(curves):
        if i == n - 1 and closed:
            cmds += 'C %f %f %f %f %f %fz ' % (*bez[1], *bez[2], *bez[3])
        else:
            cmds += 'C %f %f %f %f %f %f ' % (*bez[1], *bez[2], *bez[3])
    return cmds


def count_cp(file_name, font_name):
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(file_name)
    p_counter = 0
    for path in shapes:
        p_counter += path.points.shape[0]
    print(f"TOTAL CP:   [{p_counter}]")
    return p_counter


def write_letter_svg(c, header, fontname, beziers, subdivision_thresh, dest_path):
    cmds = ''
    svg = header

    path = '<g><path d="'
    for C in beziers:
        if subdivision_thresh is not None:
            print('subd')
            C = bezier.subdivide_bezier_chain(C, subdivision_thresh)
        cmds += bezier_chain_to_commands(C, True)
    path += cmds + '"/>\n'
    svg += path + '</g></svg>\n'

    fname = f"{dest_path}/{fontname}_{c}.svg"
    fname = fname.replace(" ", "_")
    f = open(fname, 'w')
    f.write(svg)
    f.close()
    return fname, path


def font_string_to_svgs(dest_path, font, txt, size=30, spacing=1.0, target_control=None, subdivision_thresh=None):

    fontname = os.path.splitext(os.path.basename(font))[0]
    glyph_beziers = font_string_to_beziers(font, txt, size, spacing, merge=False, target_control=target_control)
    if not os.path.isdir(dest_path):
        os.mkdir(dest_path)
    # Compute boundig box
    points = np.vstack(sum(glyph_beziers, []))
    lt = np.min(points, axis=0)
    rb = np.max(points, axis=0)
    size = rb - lt

    sizestr = 'width="%.1f" height="%.1f"' % (size[0], size[1])
    boxstr = ' viewBox="%.1f %.1f %.1f %.1f"' % (lt[0], lt[1], size[0], size[1])
    header = '''<?xml version="1.0" encoding="utf-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:ev="http://www.w3.org/2001/xml-events" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" baseProfile="full" '''
    header += sizestr
    header += boxstr
    header += '>\n<defs/>\n'

    svg_all = header

    for i, (c, beziers) in enumerate(zip(txt, glyph_beziers)):
        print(f"==== {c} ====")
        fname, path = write_letter_svg(c, header, fontname, beziers, subdivision_thresh, dest_path)

        num_cp = count_cp(fname, fontname)
        print(num_cp)
        print(font, c)
        # Add to global svg
        svg_all += path + '</g>\n'

    # Save global svg
    svg_all += '</svg>\n'
    fname = f"{dest_path}/{fontname}_{txt}.svg"
    fname = fname.replace(" ", "_")
    f = open(fname, 'w')
    f.write(svg_all)
    f.close()


if __name__ == '__main__':

    fonts = ["KaushanScript-Regular"]
    level_of_cc = 1

    if level_of_cc == 0:
        target_cp = None

    else:
        target_cp = {"A": 120, "B": 120, "C": 100, "D": 100,
                     "E": 120, "F": 120, "G": 120, "H": 120,
                     "I": 35, "J": 80, "K": 100, "L": 80,
                     "M": 100, "N": 100, "O": 100, "P": 120,
                     "Q": 120, "R": 130, "S": 110, "T": 90,
                     "U": 100, "V": 100, "W": 100, "X": 130,
                     "Y": 120, "Z": 120,
                     "a": 120, "b": 120, "c": 100, "d": 100,
                     "e": 120, "f": 120, "g": 120, "h": 120,
                     "i": 35, "j": 80, "k": 100, "l": 80,
                     "m": 100, "n": 100, "o": 100, "p": 120,
                     "q": 120, "r": 130, "s": 110, "t": 90,
                     "u": 100, "v": 100, "w": 100, "x": 130,
                     "y": 120, "z": 120
                     }

        target_cp = {k: v * level_of_cc for k, v in target_cp.items()}

    for f in fonts:
        print(f"======= {f} =======")
        font_path = f"data/fonts/{f}.ttf"
        output_path = f"data/init"
        txt = "BUNNY"
        subdivision_thresh = None
        font_string_to_svgs(output_path, font_path, txt, target_control=target_cp,
                            subdivision_thresh=subdivision_thresh)
        normalize_letter_size(output_path, font_path, txt)

        print("DONE")




