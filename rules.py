import numpy as np
import skfuzzy as fuzz
import skfuzzy.membership as mf
import matplotlib.pyplot as plt

''' Girdiler
    Entering Rate:
        * Yolov8 person detection rate
    x1_coordinate:
        * Person box x1 coordinate
    y1_coordinate:
        * Person box y1 coordinate
    x2_coordinate:
        * Person box x2 coordinate
    y2_coordinate:
        * Person box y2 coordinate
'''
''' Bilgi Kumesi
    Mamdani Cikarim Yöntemi:
        Bulanik Girdi:
            * Detection x, y
            * Detection orani

        Bilgi Kumesi:
            * Mavi bolge
            * Kirmizi bolge
            * Detection orani

        Bulanik Cikti:
            * Kisinin hangi bölgede olduğu
'''
''' Hedefler

    Hedef1: Eğer (x1_fit_hot ve y1_fit_hot) ise kisi mavi bolgededir
    Hedef2: Eğer (x2_fit_hot ve y2_fit_hot) ise kisi kirmizi bolgededir
'''

# ------- Inputs  -------
entering_rate=np.arange(0, 101, 1)
x1_coordinate=np.arange(0, 1021, 1)
y1_coordinate=np.arange(0, 501, 1)
x2_coordinate=np.arange(0, 1021, 1)
y2_coordinate=np.arange(0, 501, 1)
detection_rate=np.arange(0, 101, 1)

# ------- Membership Functions -------
x1_coordinate_cold = mf.trapmf(x1_coordinate, [220, 230, 240, 245])
y1_coordinate_cold = mf.trapmf(y1_coordinate, [350, 360, 370, 380])
x1_coordinate_hot = mf.trapmf(x1_coordinate, [260, 280, 400, 410])
y1_coordinate_hot = mf.trapmf(y1_coordinate, [400, 420, 490, 500])

x2_coordinate_cold = mf.trapmf(x2_coordinate, [270, 275, 280, 285])
y2_coordinate_cold = mf.trapmf(y2_coordinate, [350, 360, 370, 375])
x2_coordinate_hot = mf.trapmf(x2_coordinate, [295, 310, 450, 470])
y2_coordinate_hot = mf.trapmf(y2_coordinate, [380, 395, 460, 470])

detection_rate_low = mf.trapmf(detection_rate, [0, 0, 50, 60])
detection_rate_high = mf.trapmf(detection_rate, [60, 80, 100, 100])

entering_rate_low = mf.trapmf(detection_rate, [0, 0, 50, 60])
entering_rate_high = mf.trapmf(detection_rate, [60, 80, 100, 100])

# ------- View Functions -------
def viewTables():
    fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(nrows = 1, figsize =(10, 25))

    ax0.plot(x1_coordinate, x1_coordinate_cold, 'b', linewidth = 2, label = 'cold')
    ax0.plot(x1_coordinate, x1_coordinate_hot, 'r', linewidth = 2, label = 'hot')
    ax0.set_title('x1 Coordinate')
    ax0.legend()
    ax1.plot(y1_coordinate, y1_coordinate_cold, 'b', linewidth = 2, label = 'cold')
    ax1.plot(y1_coordinate, y1_coordinate_hot, 'r', linewidth = 2, label = 'hot')
    ax1.set_title('y1 Coordinate')
    ax1.legend()

    ax2.plot(x2_coordinate, x2_coordinate_cold, 'b', linewidth = 2, label = 'cold')
    ax2.plot(x2_coordinate, x2_coordinate_hot, 'r', linewidth = 2, label = 'hot')
    ax2.set_title('x2 Coordinate')
    ax2.legend()
    ax3.plot(y2_coordinate, y2_coordinate_cold, 'b', linewidth = 2, label = 'cold')
    ax3.plot(y2_coordinate, y2_coordinate_hot, 'r', linewidth = 2, label = 'hot')
    ax3.set_title('y2 Coordinate')
    ax3.legend()

    ax4.plot(detection_rate, detection_rate_low, 'y', linewidth = 2, label = 'Low')
    ax4.plot(detection_rate, detection_rate_high, 'g', linewidth = 2, label = 'High')
    ax4.set_title('Detection Rate')
    ax4.legend()

    plt.tight_layout()
    plt.show()

# ------- Mamdani Method -------
def calculateMamdani(x1, y1, x2, y2, rate):
    x1_fit_cold = fuzz.interp_membership(x1_coordinate, x1_coordinate_cold, (x1 + x2) / 2)
    x1_fit_hot = fuzz.interp_membership(x1_coordinate, x1_coordinate_hot, (x1 + x2) / 2)
    y1_fit_cold = fuzz.interp_membership(y1_coordinate, y1_coordinate_cold, y2)
    y1_fit_hot = fuzz.interp_membership(y1_coordinate, y1_coordinate_hot, y2)

    x2_fit_cold = fuzz.interp_membership(x2_coordinate, x2_coordinate_cold, (x1 + x2) / 2)
    x2_fit_hot = fuzz.interp_membership(x2_coordinate, x2_coordinate_hot, (x1 + x2) / 2)
    y2_fit_cold = fuzz.interp_membership(y2_coordinate, y2_coordinate_cold, y2)
    y2_fit_hot = fuzz.interp_membership(y2_coordinate, y2_coordinate_hot, y2)

    detection_fit_low = fuzz.interp_membership(detection_rate, detection_rate_low, rate)
    detection_fit_high = fuzz.interp_membership(detection_rate, detection_rate_high, rate)

    # ------- Rules -------
    rule1 = np.fmin(np.fmin(np.fmin(x1_fit_cold, y1_fit_cold), detection_fit_low), entering_rate_low)
    rule2 = np.fmin(np.fmin(np.fmin(x2_fit_cold, y2_fit_cold), detection_fit_low), entering_rate_low)
    rule3 = np.fmin(np.fmin(np.fmin(x1_fit_cold, y1_fit_cold), detection_fit_high), entering_rate_high)
    rule4 = np.fmin(np.fmin(np.fmin(x2_fit_cold, y2_fit_cold), detection_fit_high), entering_rate_high)
    rule5 = np.fmin(np.fmin(np.fmin(x1_fit_hot, y1_fit_hot), detection_fit_low), entering_rate_low)
    rule6 = np.fmin(np.fmin(np.fmin(x2_fit_hot, y2_fit_hot), detection_fit_low), entering_rate_low)
    rule7 = np.fmin(np.fmin(np.fmin(x1_fit_hot, y1_fit_hot), detection_fit_high), entering_rate_high)
    rule8 = np.fmin(np.fmin(np.fmin(x2_fit_hot, y2_fit_hot), detection_fit_high), entering_rate_high)

    entering_not = np.fmax(np.fmax(np.fmax(rule1,rule2), rule3), rule4)
    entering_mid = np.fmax(np.fmax(np.fmax(rule3,rule4), rule5), rule6)
    entering_high = np.fmax(np.fmax(np.fmax(rule5,rule6), rule7), rule8)

    try:
        enteringStatus = np.fmax(np.fmax(entering_not, entering_mid), entering_high)
        defuzzified  = fuzz.defuzz(entering_rate, enteringStatus, 'centroid')
        result = fuzz.interp_membership(entering_rate, enteringStatus, defuzzified)
    except:
        result = 0

    # ------- which area are you in -------
    if result == 1.0 and x2_fit_hot == 1 and y2_fit_hot == 1: return "r"
    elif result == 1.0 and x1_fit_hot == 1 and y1_fit_hot == 1: return "b"
    else: return ""
