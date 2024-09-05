import matplotlib.pyplot as plt
import mplcursors

# Sample data
x = [0, 1, 2, 3, 4, 5]
y1 = [0, 1, 4, 9, 16, 25]
y2 = [0, 1, 2, 3, 4, 5]
y3 = [25, 16, 9, 4, 1, 0]
y4 = [5, 4, 3, 2, 1, 0]

fig, axs = plt.subplots(2, 2)

lines = []
lines.append(axs[0, 0].plot(x, y1, label='Line 1', linewidth=2)[0])
lines.append(axs[0, 0].plot(x, y2, label='Line 2', linewidth=2)[0])
lines.append(axs[0, 1].plot(x, y3, label='Line 3', linewidth=2)[0])
lines.append(axs[0, 1].plot(x, y4, label='Line 4', linewidth=2)[0])
lines.append(axs[1, 0].plot(x, y1, label='Line 5', linewidth=2)[0])
lines.append(axs[1, 0].plot(x, y2, label='Line 6', linewidth=2)[0])
lines.append(axs[1, 1].plot(x, y3, label='Line 7', linewidth=2)[0])
lines.append(axs[1, 1].plot(x, y4, label='Line 8', linewidth=2)[0])

# Add interactive cursor
cursor = mplcursors.cursor(lines, hover=False)

@cursor.connect("add")
def on_add(sel):
    line = sel.artist
    line.set_linewidth(5 if line.get_linewidth() == 2 else 2)  # Toggle line width
    plt.draw()
    update_legend()

@cursor.connect("remove")
def on_remove(sel): 
    line = sel.artist
    line.set_linewidth(2)  # Reset the line width
    plt.draw()
    update_legend()


# Custom event handling for the legend
def on_legend_click(event):
    legend = event.artist
    for line in lines:
        if line.get_label() == legend.get_label():
            line.set_linewidth(5 if line.get_linewidth() == 2 else 2)
            plt.draw()
    update_legend()

def update_legend():
    for ax in axs.flat:
        legend = ax.get_legend()
        if legend:
            for legend_line, line in zip(legend.get_lines(), ax.get_lines()):
                legend_line.set_linewidth(line.get_linewidth())

for ax in axs.flat:
    legend = ax.legend()
    for legend_line in legend.get_lines():
        legend_line.set_picker(True)


fig.canvas.mpl_connect('pick_event', on_legend_click)

plt.show()
