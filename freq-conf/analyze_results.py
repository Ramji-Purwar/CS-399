#!/usr/bin/env python3

import csv
import sys
from collections import defaultdict
import xml.etree.ElementTree as ET

def generate_svg(data, out_path="results/sweep_plot.svg"):
    # data is a dict: cores -> list of (burst_billions, freq_ghz)
    if not data:
        print("No data to plot.")
        return

    width = 900
    height = 500
    margin = {"top": 40, "right": 150, "bottom": 60, "left": 80}
    plot_width = width - margin["left"] - margin["right"]
    plot_height = height - margin["top"] - margin["bottom"]

    # Find max/min
    all_bursts = []
    all_freqs = []
    for cores, pts in data.items():
        for b, f in pts:
            all_bursts.append(b)
            all_freqs.append(f)

    max_x = max(all_bursts) if all_bursts else 50.0
    min_x = 0
    max_y = max(all_freqs) if all_freqs else 4.0
    # Add a little headroom for max_y
    max_y = ((int(max_y * 10) + 2) / 10.0) 
    min_y = 0.0

    def x_to_svg(x):
        return margin["left"] + (x - min_x) / (max_x - min_x) * plot_width

    def y_to_svg(y):
        return margin["top"] + plot_height - ((y - min_y) / (max_y - min_y) * plot_height)

    svg = ET.Element("svg", width=str(width), height=str(height), xmlns="http://www.w3.org/2000/svg")
    
    # Background
    ET.SubElement(svg, "rect", width="100%", height="100%", fill="#ffffff")
    
    # Axes
    ET.SubElement(svg, "line", x1=str(margin["left"]), y1=str(margin["top"] + plot_height),
                  x2=str(margin["left"] + plot_width), y2=str(margin["top"] + plot_height),
                  stroke="black", stroke_width="2")
    ET.SubElement(svg, "line", x1=str(margin["left"]), y1=str(margin["top"]),
                  x2=str(margin["left"]), y2=str(margin["top"] + plot_height),
                  stroke="black", stroke_width="2")

    # X-axis ticks (every 5B)
    for x_tick in range(0, int(max_x) + 1, max(1, int(max_x)//10)):
        sx = x_to_svg(x_tick)
        sy = margin["top"] + plot_height
        ET.SubElement(svg, "line", x1=str(sx), y1=str(sy), x2=str(sx), y2=str(sy+5), stroke="black")
        lbl = ET.SubElement(svg, "text", x=str(sx), y=str(sy+20), text_anchor="middle", font_family="sans-serif", font_size="12")
        lbl.text = str(x_tick)

    # Y-axis ticks (every 0.5 GHz)
    y_tick = 0.0
    while y_tick <= max_y:
        sy = y_to_svg(y_tick)
        sx = margin["left"]
        ET.SubElement(svg, "line", x1=str(sx-5), y1=str(sy), x2=str(sx), y2=str(sy), stroke="black")
        lbl = ET.SubElement(svg, "text", x=str(sx-10), y=str(sy+4), text_anchor="end", font_family="sans-serif", font_size="12")
        lbl.text = f"{y_tick:.1f}"
        
        # Grid line
        ET.SubElement(svg, "line", x1=str(sx), y1=str(sy), x2=str(sx+plot_width), y2=str(sy), stroke="#eeeeee")
        y_tick += 0.5

    # Labels
    title = ET.SubElement(svg, "text", x=str(width/2), y="25", text_anchor="middle", font_family="sans-serif", font_size="16", font_weight="bold")
    title.text = "AVX-512 Frequency Throttling vs Burst Size"

    x_label = ET.SubElement(svg, "text", x=str(margin["left"] + plot_width/2), y=str(height - 15), text_anchor="middle", font_family="sans-serif", font_size="14")
    x_label.text = "Burst Size (Billions of AVX-512 FMA Instructions)"

    y_label = ET.SubElement(svg, "text", x="25", y=str(margin["top"] + plot_height/2), text_anchor="middle", font_family="sans-serif", font_size="14", transform=f"rotate(-90 25 {margin['top'] + plot_height/2})")
    y_label.text = "Frequency (GHz)"

    # Plot data
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    
    legend_y = margin["top"]
    sorted_cores = sorted(data.keys())
    
    for i, cores in enumerate(sorted_cores):
        pts = sorted(data[cores])
        color = colors[i % len(colors)]
        
        # Draw line
        path_d = []
        for j, (b, f) in enumerate(pts):
            cmd = "M" if j == 0 else "L"
            path_d.append(f"{cmd} {x_to_svg(b)} {y_to_svg(f)}")
        
        path = ET.SubElement(svg, "path", d=" ".join(path_d), fill="none", stroke=color, stroke_width="2")
        
        # Draw dots
        for b, f in pts:
            ET.SubElement(svg, "circle", cx=str(x_to_svg(b)), cy=str(y_to_svg(f)), r="3", fill=color)
            
        # Legend
        lx = margin["left"] + plot_width + 20
        ly = legend_y + i * 25
        ET.SubElement(svg, "rect", x=str(lx), y=str(ly-5), width="15", height="10", fill=color)
        llbl = ET.SubElement(svg, "text", x=str(lx+25), y=str(ly+4), font_family="sans-serif", font_size="12")
        llbl.text = f"{cores} Core{'s' if cores>1 else ''}"

    tree = ET.ElementTree(svg)
    tree.write(out_path)
    print(f"Generated plot: {out_path}")

def main():
    csv_file = "results/sweep_data.csv"
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]

    data = defaultdict(list)
    try:
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                c = int(row["active_cores"])
                b = float(row["burst_billions"])
                fq = float(row["freq_ghz"])
                data[c].append((b, fq))
    except Exception as e:
        print(f"Failed to read CSV {csv_file}: {e}")
        return

    generate_svg(data)

if __name__ == "__main__":
    main()
