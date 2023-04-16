import matplotlib.pyplot as plt #Plotting functionality
import matplotlib.patches as patch #For plotting BMD and SFD
import seaborn as sns

plt.style.use('seaborn')


def PlotNineStructures(n1, n2, n3, n4, n5, n6, n7, n8, n9, label_offset=0.005, vert_margin=0.01, hor_margin=0.01):
    dx = label_offset
    dy = label_offset

    fig = plt.figure(figsize=(12, 12))

    axs = [fig.add_subplot(3, 3, i + 1) for i in range(9)]
    nodes = [n1, n2, n3, n4, n5, n6, n7, n8, n9]
    fig.suptitle('9 Random Structures from Brute Force Generator')
    fig.supxlabel('Distance (m)')
    fig.supylabel('Distance (m)')

    for i in range(0, len(axs)):
        # Plot members
        for mbr in members:
            node_i = mbr[0]  # Node number for node i of this member
            node_j = mbr[1]  # Node number for node j of this member

            ix = nodes[i][node_i - 1, 0]  # x-coord of node i of this member
            iy = nodes[i][node_i - 1, 1]  # y-coord of node i of this member
            jx = nodes[i][node_j - 1, 0]  # x-coord of node j of this member
            jy = nodes[i][node_j - 1, 1]  # y-coord of node j of this member

            axs[i].plot([ix, jx], [iy, jy], 'b')  # Member

        # Plot nodes
        # for n,node in enumerate(nodes[i]):
        #     axs[i].plot([node[0]],[node[1]],'bo')
        #     label = str(n+1)
        #     axs[i].text(node[0]+dx, node[1]+dy, label, fontsize = 10)

        # Set axis limits to provide margin around structure
        maxX = nodes[i].max(0)[0]
        maxY = nodes[i].max(0)[1]
        minX = nodes[i].min(0)[0]
        minY = nodes[i].min(0)[1]
        axs[i].set_xlim([minX - hor_margin, maxX + hor_margin])
        axs[i].set_ylim([minY - vert_margin, maxY + vert_margin])
        axs[i].set_title('Structure {}'.format(i + 1))
        axs[i].grid()
        # axs[i].set_aspect(1.4)
    fig.subplots_adjust(wspace=0.25, hspace=0.3)