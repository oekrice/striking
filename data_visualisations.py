'''
Copyright (C) 2025 Oliver Rice - All Rights Reserved

Permission is hereby granted to any individual to use and modify this software solely for personal, non-commercial purposes.

You May Not:

 - Distribute, sublicense, or share the software or modified versions in any form.

 - Use the software or any part of it for commercial purposes.

 - Use the software as part of a service, product, or offering to others.

This software is provided "as is", without warranty of any kind, express or implied. In no event shall the authors be liable for any claim, damages, or other liability.

If you would like to license or publish this software commerically, please contact oliverricesolar@gmail.com
'''

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
#This file contains all the stuff which is currently cluttering up the second page, which has got a bit out of hand

cmap = plt.cm.gnuplot2
cmap = [
    '#1f77b4',  
    '#ff7f0e', 
    '#2ca02c',  
    '#d62728', 
    '#9467bd', 
    '#8c564b', 
    '#e377c2',  
    '#7f7f7f', 
    '#bcbd22',  
    '#17becf'  
]

def calculate_stats(Strike_Data):
    #Do stats now
    Strike_Data.alldiags = np.zeros((3,3,Strike_Data.nbells))   #Type, stroke, bell
    diffs = np.array(Strike_Data.raw_actuals)[1:] - np.array(Strike_Data.raw_actuals)[:-1]
    Strike_Data.cadence = np.mean(diffs)*(2*Strike_Data.nbells)/(2*Strike_Data.nbells + 1)   #Mean interbell gap (ish)

    Strike_Data.time_errors = np.zeros((Strike_Data.nbells, int(len(np.array(Strike_Data.raw_actuals))/Strike_Data.nbells)))   #Errors through time for the whole touch
    Strike_Data.lead_times = np.zeros((Strike_Data.time_errors.shape), dtype = bool)
    for plot_id in range(3):
        for bell in range(1,Strike_Data.nbells+1):#nbells):
            
            bellstrikes = np.where(Strike_Data.raw_bells == bell)[0]
            targetstrikes = np.where(Strike_Data.correct_bells == bell)[0]

            #Find the times at which bellstrikes are leading, as tru false array
            lead_bell = [True if val in np.arange(0,len(bellstrikes)*Strike_Data.nbells,Strike_Data.nbells) else False for val in bellstrikes]
            Strike_Data.lead_times[bell-1] = lead_bell

            if Strike_Data.use_method_info:
                Strike_Data.errors = np.array(Strike_Data.raw_actuals[bellstrikes] - Strike_Data.raw_target[targetstrikes])
            else:
                Strike_Data.errors = np.array(Strike_Data.raw_actuals[bellstrikes] - Strike_Data.raw_target[bellstrikes])

            confs = np.array(Strike_Data.raw_data["Confidence"][bellstrikes])
            
            Strike_Data.time_errors[bell-1, :] = Strike_Data.errors
            if Strike_Data.remove_confidence:
                Strike_Data.time_errors[bell-1, confs < 0.9] = 0.0

            bellstrikes = bellstrikes[bellstrikes/Strike_Data.nbells >= Strike_Data.min_include_change]
            bellstrikes = bellstrikes[bellstrikes/Strike_Data.nbells <= Strike_Data.max_include_change]
                
            if len(bellstrikes) < 2:
                st.error('Increase range -- stats are all wrong')
                st.stop()
                
            Strike_Data.errors = np.array(Strike_Data.raw_actuals[bellstrikes] - Strike_Data.raw_target[bellstrikes])
            confs = np.array(Strike_Data.raw_data["Confidence"][bellstrikes])
            
            #Attempt to remove outliers (presumably method mistakes, hawkear being silly or other spannering)
            maxlim = Strike_Data.cadence*0.75
            minlim = -Strike_Data.cadence*0.75

            #Trim for the appropriate stroke
            if plot_id == 1:
                Strike_Data.errors = Strike_Data.errors[::2]
                confs = confs[::2]
            if plot_id == 2:
                Strike_Data.errors = Strike_Data.errors[1::2]
                confs = confs[1::2]
            count = len(Strike_Data.errors)

            if Strike_Data.remove_confidence:
                Strike_Data.errors[confs < 0.9] = 0.0
                count -= np.sum(confs < 0.9)
            
            if Strike_Data.remove_mistakes:
                #Adjust stats to disregard these properly
                count -= np.sum(Strike_Data.errors > maxlim)
                count -= np.sum(Strike_Data.errors < minlim)

                Strike_Data.errors[Strike_Data.errors > maxlim] = 0.0
                Strike_Data.errors[Strike_Data.errors < minlim] = 0.0

            if count > 2:
            #Diagnostics
                Strike_Data.alldiags[0,plot_id,bell-1] = np.sum(Strike_Data.errors)/count
                Strike_Data.alldiags[1,plot_id,bell-1] = np.sqrt(np.sum((Strike_Data.errors-np.sum(Strike_Data.errors)/count)**2)/count)
                Strike_Data.alldiags[2,plot_id,bell-1] = np.sqrt(np.sum(Strike_Data.errors**2)/count)
    return Strike_Data

#Do plain-text appraisal of the striking. Give the most accurate and most consistent bells, and oddstruckness. Will add leading if that's a thing too
@st.cache_data
def obtain_striking_markdown(alldiags, time_errors, lead_times, cadence, remove_mistakes):
    nbells = np.shape(time_errors)[0]
    bell_consistencies = alldiags[1,0,:]
    bell_accuracies = alldiags[2,0,:]

    most_consistent = np.where(bell_consistencies == np.min(bell_consistencies))[0][0]
    most_accurate = np.where(bell_accuracies == np.min(bell_accuracies))[0][0]

    lines = []
    lines.append('<u>Striking Report</u>: <br>')
    if most_consistent == most_accurate:
        lines.append('Bell %d was the most consistent and accurate <br>' % (most_consistent + 1))
    else:
        lines.append('Bell %d was the most accurate <br>' % (most_accurate + 1))
        lines.append('Bell %d was the most consistent <br>' % (most_consistent + 1))

    lines.append('  <br>')
    maxlim = cadence*0.75
    minlim = -cadence*0.75

    threshold_1 = 0.7; threshold_2 = 0.8
    overall_oddstrucknesses = np.zeros((nbells, 3), dtype = int)

    def number_to_words(number):
        if number == -2:
            return 'consistently quick'
        elif number == -1:
            return 'quick'
        elif number == 2:
            return 'consistently slow'
        elif number == 1:
            return 'slow'
        else:
            return ''

    def produce_strings(bell, lines, errors_all, errors_hand, errors_back, lead = False):

        if len(errors_all) < 20: #Not enough stats to go off
            return lines
        all_prop = np.sum(errors_all > 0)/len(errors_all)
        hand_prop = np.sum(errors_hand > 0)/len(errors_hand)
        back_prop = np.sum(errors_back > 0)/len(errors_back)
        if all_prop > threshold_2:
            overall_oddstrucknesses[bell,0] = 2
        elif all_prop > threshold_1:
            overall_oddstrucknesses[bell,0] = 1
        elif all_prop < 1.0-threshold_2:
            overall_oddstrucknesses[bell,0] = -2
        elif all_prop < 1.0-threshold_1:
            overall_oddstrucknesses[bell,0] = -1

        if hand_prop > threshold_2:
            overall_oddstrucknesses[bell,1] = 2
        elif hand_prop > threshold_1:
            overall_oddstrucknesses[bell,1] = 1
        elif hand_prop < 1.0-threshold_2:
            overall_oddstrucknesses[bell,1] = -2
        elif hand_prop < 1.0-threshold_1:
            overall_oddstrucknesses[bell,1] = -1

        if back_prop > threshold_2:
            overall_oddstrucknesses[bell,2] = 2
        elif back_prop > threshold_1:
            overall_oddstrucknesses[bell,2] = 1
        elif back_prop < 1.0-threshold_2:
            overall_oddstrucknesses[bell,2] = -2
        elif back_prop < 1.0-threshold_1:
            overall_oddstrucknesses[bell,2] = -1
    
        #Figure out a way of writing all the things consistently and grammatically nice
        if (overall_oddstrucknesses[bell][1] == overall_oddstrucknesses[bell][2]) and overall_oddstrucknesses[bell][1] != 0:
            text = number_to_words(overall_oddstrucknesses[bell][1])
            if not lead:
                lines.append('Bell %d was generally %s. <br>' % (bell+1, text))
            else:
                lines.append('Bell %d was generally %s leading. <br>' % (bell+1, text))
        else:
            text1 = None; text2 = None
            if overall_oddstrucknesses[bell][1] != 0:
                text1 = number_to_words(overall_oddstrucknesses[bell][1])
            if overall_oddstrucknesses[bell][2] != 0:
                text2 = number_to_words(overall_oddstrucknesses[bell][2])

            if text1 is not None:
                if not lead:
                    lines.append('Bell %d was %s at handstroke <br>' % (bell+1, text1))
                else:
                    lines.append('Bell %d was %s leading at handstroke<br>' % (bell+1, text1))
            if text2 is not None:
                if not lead:
                    lines.append('Bell %d was %s at backstroke <br>' % (bell+1, text2))
                else:
                    lines.append('Bell %d was %s leading at backstroke<br>' % (bell+1, text2))
        return lines

    #Filter out leading blows. Bit tricky...
    #Determine confidence intervals
    for bell in range(nbells):
        errors_all = time_errors[bell,:]
        hand_leads = lead_times[bell,:-1:2]
        back_leads = lead_times[bell,1::2]
        leading_all = time_errors[bell][lead_times[bell]]

        if remove_mistakes:
            #Adjust stats to disregard these properly
            errors_all[errors_all > maxlim] = 0.0
            errors_all[errors_all < minlim] = 0.0

        errors_hand = time_errors[bell,:-1:2]
        errors_back = time_errors[bell,1::2]

        leading_hand = errors_hand[hand_leads]
        leading_back = errors_back[back_leads]

        lines = produce_strings(bell, lines, errors_all, errors_hand, errors_back, lead = False)
        lines = produce_strings(bell, lines, leading_all, leading_hand, leading_back, lead = True)

    markdown_html = '<pre>' +  ' '.join(lines) + '</pre>'
    st.html(markdown_html)
            
    return

#Want error through time for each bell, and do as a snazzy plot?
@st.cache_data
def plot_errors_time(time_errors, min_plot_change, max_plot_change, absvalues, highlight_bells, strokes_plot, smooth):
    #Just do both strokes to start with because easy
    fig5, ax5 = plt.subplots(1, figsize = (10,5))
    all_xs = np.arange(min_plot_change, max_plot_change)
    hand_xs = np.arange(min_plot_change, max_plot_change, 2)
    back_xs = np.arange(min_plot_change + 1, max_plot_change, 2)
    #Just calculate everything. Doesn't take long...
    if absvalues == "Absolute Error":
        time_errors = np.abs(time_errors[:,min_plot_change:max_plot_change])
    else:  
        time_errors = time_errors[:,min_plot_change:max_plot_change]
    
    all_ys = time_errors[:,:]
    hand_ys = time_errors[:,::2]
    back_ys = time_errors[:,1::2]
    
    if smooth:
        all_ys = gaussian_filter1d(all_ys, 6, axis = 1)
        hand_ys = gaussian_filter1d(hand_ys, 3, axis = 1)
        back_ys = gaussian_filter1d(back_ys, 3, axis = 1)
        
    avg_ys = np.mean(all_ys, axis = 0)
    hand_avg_ys = np.mean(hand_ys, axis = 0)
    back_avg_ys = np.mean(back_ys, axis = 0)
    
    if len(strokes_plot) > 1:
        mode = 2  #Multiple strokes, one bell
    else:
        mode = 1  #Multiple bells, one stroke
    
    for bell in highlight_bells:
        for stroke in strokes_plot:
            if bell == "Average":
                if stroke == "Both Strokes":
                    if mode == 2:
                        ax5.plot(all_xs, avg_ys, c = 'gray', label = 'Both Strokes')
                    else:
                        ax5.plot(all_xs, avg_ys, c = 'black', linewidth = 3, zorder = 10, label = bell)
                if stroke == "Handstrokes":
                    if mode == 2:
                        ax5.plot(hand_xs, hand_avg_ys, c = 'red', label = 'Handstrokes')
                    else:
                        ax5.plot(hand_xs, hand_avg_ys, c = 'black', linewidth = 3, zorder = 10, label = bell)
                if stroke == "Backstrokes":
                    if mode == 2:
                        ax5.plot(back_xs, back_avg_ys, c = 'green', label = 'Backstrokes')
                    else:
                        ax5.plot(back_xs, back_avg_ys, c = 'black', linewidth = 3, zorder = 10, label = bell)
            else:
                if stroke == "Both Strokes":
                    if mode == 2:
                        ax5.plot(all_xs, all_ys[int(bell[5:])-1], c = 'gray', label = 'Both Strokes')
                    else:
                        ax5.plot(all_xs, all_ys[int(bell[5:])-1], c = cmap[(int(bell[5:])-1)%10], label = bell)
                if stroke == "Handstrokes":
                    if mode == 2:
                        ax5.plot(hand_xs, hand_ys[int(bell[5:])-1], c = 'red', label = 'Handstrokes')
                    else:
                        ax5.plot(hand_xs, hand_ys[int(bell[5:])-1], c = cmap[(int(bell[5:])-1)%10], label = bell)
                if stroke == "Backstrokes":
                    if mode == 2:
                        ax5.plot(back_xs, back_ys[int(bell[5:])-1], c = 'green', label = 'Backstrokes')
                    else:
                        ax5.plot(back_xs, back_ys[int(bell[5:])-1], c = cmap[(int(bell[5:])-1)%10], label = bell)
    
    plt.xlabel('Change Number')
    plt.ylabel('Error (ms)')
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig5)
    plt.clf()
    plt.close()

    return

@st.cache_data
def plot_bar_charts(alldiags, nbells, titles):
    fig3, axs3 = plt.subplots(3, figsize = (12,8))
    bar_width = 0.3

    data_titles = ['Avg. Error (positive is slow, negative is quick)', 'Std. Dev. from Average', 'Std. Dev. From Ideal']

    x = np.arange(nbells)
    for plot_id in range(3):
        ax = axs3[plot_id]

        yrange = np.max(alldiags[plot_id,:,:]) - np.min(alldiags[plot_id,:,:])
        ymin = np.min(alldiags[plot_id,:,:]) - yrange*0.2
        ymax = np.max(alldiags[plot_id,:,:]) + yrange*0.2
        
        rects0 = ax.bar(x-bar_width*1,alldiags[plot_id,0,:],bar_width,label = titles[0], color='lightgray')
        ax.bar_label(rects0, padding = 3, fmt = '%d')

        rects1 = ax.bar(x-bar_width*0,alldiags[plot_id,1,:],bar_width,label = titles[1], color='red')
        ax.bar_label(rects1, padding = 3, fmt = '%d')

        rects2 = ax.bar(x+bar_width*1.0,alldiags[plot_id,2,:],bar_width,label = titles[2], color='blue')
        ax.bar_label(rects2, padding = 3, fmt = '%d')

        ax.set_xticks(np.arange(nbells), np.arange(1,nbells+1))
        ax.set_title(data_titles[plot_id])
        ax.set_ylim(ymin, ymax)

        if plot_id == 0:
            ax.legend()
            
    plt.tight_layout()
    st.pyplot(fig3)
    plt.clf()
    plt.close()

@st.cache_data
def plot_histograms(errors, x_range, nbins, nbells, raw_bells, correct_bells, min_include_change, max_include_change, use_method_info, remove_mistakes, cadence, raw_actuals, raw_target, titles):

    for plot_id in range(3):
        #Everything, then handstrokes, then backstrokes
        nrows = max(2,(nbells)//3)
        ncols = int((nbells-1e-6)/nrows) + 1
        if nrows*ncols < nbells:
            ncols += 1
        fig2, axs2 = plt.subplots(nrows,ncols, figsize = (10,7))
        for bell in range(1,nbells+1):#nbells):
            #Extract data for this bell

            bellstrikes = np.where(raw_bells == bell)[0]
            targetstrikes = np.where(correct_bells == bell)[0]

            bellstrikes = bellstrikes[bellstrikes/nbells >= min_include_change]
            bellstrikes = bellstrikes[bellstrikes/nbells <= max_include_change]

            targetstrikes = targetstrikes[targetstrikes/nbells >= min_include_change]
            targetstrikes = targetstrikes[targetstrikes/nbells <= max_include_change]

            if len(bellstrikes) < 2:
                st.error('Increase range -- stats are all wrong')
                st.stop()

            if use_method_info:
                errors = np.array(raw_actuals[bellstrikes] - raw_target[targetstrikes])
            else:
                errors = np.array(raw_actuals[bellstrikes] - raw_target[bellstrikes])

            #Attempt to remove outliers (presumably method mistakes, hawkear being silly or other spannering)
            maxlim = cadence*0.75
            minlim = -cadence*0.75

            #Trim for the appropriate stroke
            if plot_id == 1:
                errors = errors[::2]
            if plot_id == 2:
                errors = errors[1::2]

            count = len(errors)

            if remove_mistakes:
                #Adjust stats to disregard these properly
                count -= np.sum(errors > maxlim)
                count -= np.sum(errors < minlim)

                errors[errors > maxlim] = 0.0
                errors[errors < minlim] = 0.0

            ax = axs2[(bell-1)//ncols, (bell-1)%ncols]

            ax.set_title('Bell %d' % bell)
            bin_bounds = np.linspace(-x_range, x_range, nbins+1)
            n, bins, _ = ax.hist(errors, bins = bin_bounds)

            curve = gaussian_filter1d(n, sigma = nbins/20)
            ax.plot(0.5*(bins[1:] + bins[:-1]),curve, c= 'black')
            ax.set_xlim(-x_range, x_range)
            ax.set_ylim(0,np.max(n))
            ax.plot([0,0],[0,max(n)], linewidth = 2)
            ax.set_yticks([])
        plt.suptitle(titles[plot_id])
        plt.tight_layout()
        st.pyplot(fig2)
        plt.clf()
        plt.close()

@st.cache_data
def plot_boxes(errors, nbells, titles):
    fig4, axs4 = plt.subplots(3, figsize = (12,8))
    for plot_id in range(3):
        #Everything, then handstrokes, then backstrokes

        ax = axs4[plot_id]

        
        for bell in range(1,nbells+1):#nbells):
            #Extract data for this bell

            bell_errors = errors[bell-1]

            #Trim for the appropriate stroke
            if plot_id == 1:
                bell_errors = bell_errors[::2]
            if plot_id == 2:
                bell_errors = bell_errors[1::2]

            #Box plot data
            ax.boxplot(bell_errors,positions = [bell], sym = 'x', widths = 0.35, zorder = 1)
        #ax.axhline(0.0, c = 'black', linestyle = 'dashed')

        ax.set_ylim(-150,150)
        ax.set_title(titles[plot_id])

    plt.tight_layout()
    st.pyplot(fig4)
    plt.clf()
    plt.close()

@st.cache_data
def plot_blue_line(raw_target_plot, raw_actuals, raw_bells, nbells, lead_length, min_plot_change, max_plot_change, highlight_bells, view_numbers = False):
    
    bell_names = ['1','2','3','4','5','6','7','8','9','0','E','T','A','B','C','D']
    highlight_bells = [int(highlight_bells[val][5:]) for val in range(len(highlight_bells))]

    nrows = int(len(raw_actuals)/nbells)

    toprint = []
    orders = []; starts = []; ends = []
    for row in range(nrows):
        actual = np.array(raw_actuals[row*nbells:(row+1)*nbells])
        target = np.array(raw_target_plot[row*nbells:(row+1)*nbells])
        bells =  np.array(raw_bells[row*nbells:(row+1)*nbells])  
        toprint.append(actual-target)
        orders.append(bells)
        starts.append(np.min(target))
        ends.append(np.max(target))

    nrows_plot = max_plot_change - min_plot_change
    rows_per_plot = 61#6*int(nrows_plot//24)
    if nbells < 9:
        nplotsk = max(3, nrows_plot//rows_per_plot + 1)
    else:
        nplotsk = max(2, nrows_plot//rows_per_plot + 1)
    nplotsk = min(nplotsk, 3)
    min_rows_per_plot = int(nrows_plot/nplotsk) + 2
    rows_per_plot = int(np.ceil(min_rows_per_plot / lead_length) * lead_length)

    nplotsk = int(np.ceil((nrows_plot-1)/rows_per_plot))

    #fig,axs = plt.subplots(1,ncols, figsize = (15,4*nrows/(nbells + 4)))
    fig1, axs1 = plt.subplots(1,nplotsk, figsize = (10, 100))
    for plot in range(nplotsk):
        
        if nplotsk > 1:
            ax = axs1[plot]
        else:
            ax = axs1
        maxrow = max_plot_change
        for bell in range(1,nbells+1):#nbells):

            if bell in highlight_bells:
                points = []  ; changes = []
                bellstrikes = np.where(raw_bells == bell)[0]
                for row in range(plot*rows_per_plot+ min_plot_change, min((plot+1)*rows_per_plot + min_plot_change + 1, maxrow)):
                    #Find linear position... Linear interpolate?
                    target_row = np.array(raw_target_plot[row*nbells:(row+1)*nbells])
                    if len(target_row) == nbells:
                        ys = np.arange(1,nbells+1)
                        f = interpolate.interp1d(target_row, ys, fill_value = "extrapolate")
                        rat = float(f(raw_actuals[bellstrikes].tolist()[row]))
                        points.append(rat); changes.append(row)
                        
                        if view_numbers:
                            ax.text(rat, row, bell_names[bell-1], horizontalalignment = 'center', verticalalignment = 'center')

                if len(highlight_bells) > 0:
                    ax.plot(points, changes,label = bell, c = cmap[(bell-1)%10], linewidth = 2)
                else:
                    ax.plot(points, changes,label = bell, c = cmap[(bell-1)%10])
            else:
                points = []  ; changes = []
                bellstrikes = np.where(raw_bells == bell)[0]
                for row in range(plot*rows_per_plot+ min_plot_change, min((plot+1)*rows_per_plot + min_plot_change + 1, maxrow)):
                    #Find linear position... Linear interpolate?
                    target_row = np.array(raw_target_plot[row*nbells:(row+1)*nbells])
                    if len(target_row) == nbells:
                        ys = np.arange(1,nbells+1)
                        f = interpolate.interp1d(target_row, ys, fill_value = "extrapolate")
                        rat = float(f(raw_actuals[bellstrikes].tolist()[row]))
                        points.append(rat); changes.append(row)
                        if view_numbers:
                            ax.text(rat, row, bell_names[bell-1], horizontalalignment = 'center', verticalalignment = 'center')
                            
                if len(highlight_bells) > 0:
                    ax.plot(points, changes,label = bell, c = 'grey', linewidth  = 0.5)
                else:
                    if view_numbers:
                        ax.plot(points, changes,label = bell, c = cmap[(bell-1)%10], linewidth = 0.5)
                    else:
                        ax.plot(points, changes,label = bell, c = cmap[(bell-1)%10])

                
            ax.plot((bell)*np.ones(len(points)), changes, c = 'black', linewidth = 0.5, linestyle = 'dotted', zorder = 0)
            
        row_guides = [np.column_stack([np.arange(-1,nbells+3), row*np.ones(nbells+4)]) for row in range(plot*rows_per_plot+ min_plot_change, (plot+1)*rows_per_plot + min_plot_change + 1)]
        #for row in range(min_plot_change, max_plot_change):
        #    ax.plot(np.arange(-1,nbells+3), row*np.ones(nbells+4), c = 'black', linewidth = 0.5, linestyle = 'dotted', zorder = 0)
        line_collection = LineCollection(row_guides, color = 'black', linewidth = 0.5, linestyle = 'dotted', zorder = 0)
        ax.add_collection(line_collection)
        
        plt.gca().invert_yaxis()
        ax.set_ylim((plot+1)*rows_per_plot + min_plot_change, plot*rows_per_plot+ min_plot_change )
        ax.set_xlim(-1,nbells+2)
        ax.set_xticks([])
        ax.set_aspect('equal')
        #if plot == nplotsk-1:
        #    plt.legend()
        #ax.set_yticks([])
    plt.tight_layout()
    st.pyplot(fig1)
    plt.clf()
    plt.close()
