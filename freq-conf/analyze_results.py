#!/usr/bin/env python3

import csv
import sys
from collections import defaultdict
import xml.etree.ElementTree as ET

def generate_svg(data, out_path="results/cores_plot.svg"):
    # data is a dict: cores -> list of frequencies
    if not data:
        print("No data to plot.")
        return

    # Compute average frequency for each core count
    avg_freqs = {}
    for cores, freqs in data.items():
        avg_freqs[cores] = sum(freqs) / len(freqs)

    width = 900
    height = 500
    margin = {"top": 40, "right": 40, "bottom": 60, "left": 80}
    plot_width = width - margin["left"] - margin["right"]
    plot_height = height - margin["top"] - margin["bottom"]

    sorted_cores = sorted(avg_freqs.keys())
    
    max_x = 28
    min_x = 0
    
    all_freqs = list(avg_freqs.values())
    max_y = 4.0
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

    # X-axis ticks (every 2 cores)
    for x_tick in range(0, max_x + 1, 2):
        sx = x_to_svg(x_tick)
        sy = margin["top"] + plot_height
        ET.SubElement(svg, "line", x1=str(sx), y1=str(sy), x2=str(sx), y2=str(sy+5), stroke="black")
        if x_tick > 0:
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
    title.text = "All-Core AVX-512 Turbo Frequency vs Number of Active Cores"

    x_label = ET.SubElement(svg, "text", x=str(margin["left"] + plot_width/2), y=str(height - 15), text_anchor="middle", font_family="sans-serif", font_size="14")
    x_label.text = "Number of Active Cores"

    y_label = ET.SubElement(svg, "text", x="25", y=str(margin["top"] + plot_height/2), text_anchor="middle", font_family="sans-serif", font_size="14", transform=f"rotate(-90 25 {margin['top'] + plot_height/2})")
    y_label.text = "Average Frequency (GHz)"

    # Plot data (Draw line)
    color = "#d62728"
    path_d = []
    for j, cores in enumerate(sorted_cores):
        f = avg_freqs[cores]
        cmd = "M" if j == 0 else "L"
        path_d.append(f"{cmd} {x_to_svg(cores)} {y_to_svg(f)}")
    
    if path_d:
        ET.SubElement(svg, "path", d=" ".join(path_d), fill="none", stroke=color, stroke_width="3")
    
    # Draw dots
    for cores in sorted_cores:
        f = avg_freqs[cores]
        ET.SubElement(svg, "circle", cx=str(x_to_svg(cores)), cy=str(y_to_svg(f)), r="5", fill=color)

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
                # We ignore burst size and just collect all frequencies for this core count
                # Optional: exclude the 0.0B burst if you want strictly active AVX-512 measurements
                b = float(row["burst_billions"])
                if b > 0:
                    fq = float(row["freq_ghz"])
                    data[c].append(fq)
    except Exception as e:
        print(f"Failed to read CSV {csv_file}: {e}")
        return

    generate_svg(data)

if __name__ == "__main__":
    main()
