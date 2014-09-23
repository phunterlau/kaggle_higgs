'''
original nice interface work from https://github.com/ghl3/higgs-kaggle 
I rewrote with pandas, and added many other features
some of them are good, some of them are....tricky
'''
import pandas

import numpy as np

import math
import math
from math import sin, cos, sinh


def jet_partition(row):
    jet_num = row['PRI_jet_num']
    if jet_num==0:
        return 'zero_jet'
    elif jet_num==1:
        return 'one_jet'
    else:
        return 'multi_jet'

#categorize num of jets
def is_0_jet(row):
    frame = row['PRI_jet_num'].map(lambda x: 1 if x==0 else 0)
    frame.name = 'IS_zero_jet'
    return frame

def is_1_jet(row):
    frame = row['PRI_jet_num'].map(lambda x: 1 if x==1 else 0)
    frame.name = 'IS_one_jet'
    return frame

def is_2_jet(row):
    frame = row['PRI_jet_num'].map(lambda x: 1 if x==2 else 0)
    frame.name = 'IS_di_jet'
    return frame

def is_multi_jet(row):
    frame = row['PRI_jet_num'].map(lambda x: 1 if x>2 else 0)
    frame.name = 'IS_multi_jet'
    return frame

jet_num_feat = [is_0_jet,is_1_jet,is_2_jet,is_multi_jet]

# Momentum

def px(pt, eta, phi): 
    frame = pt * np.cos(phi)
    mask = (pt<0)
    frame[mask]=-999.0
    return frame

def py(pt, eta, phi):
    frame = pt * np.sin(phi)
    mask = (pt<0)
    frame[mask]=-999.0
    return frame

def pz(pt, eta, phi):
    frame = pt * np.sinh(eta)
    #pt * np.tan(eta.map(lambda x: 2*math.arctan(math.exp(-x))))
    mask = (pt<0)
    frame[mask]=-999.0
    return frame


def p_tot(pt, eta, phi):
    x = px(pt, eta, phi)
    y = py(pt, eta, phi)
    z = pz(pt, eta, phi)
    frame = np.sqrt(x*x + y*y + z*z)
    mask = (pt<0)
    frame[mask]=-999.0
    return frame

def _calculate_momenta(df, prefix):
    pt = df[prefix+'pt']
    eta = df[prefix+'eta']
    phi = df[prefix+'phi']

    return pandas.DataFrame({prefix+'px' : px(pt, eta, phi),
                      prefix+'py' : py(pt, eta, phi),
                      prefix+'pz' : pz(pt, eta, phi),
                      prefix+'p_tot' : p_tot(pt, eta, phi)})

#abs values of these px py pz can have separation
def _abs_(df,prefix):
    frame = df[prefix].map(lambda x: abs(x))
    frame.name = prefix+'_abs'
    return frame

def _get_abs_momenta(df,prefix):
    return pandas.DataFrame({prefix+'px_abs' : _abs_(df,prefix+'px'),
                      prefix+'py_abs' : _abs_(df,prefix+'py'),
                      prefix+'pz_abs' : _abs_(df,prefix+'pz'),
                      })

def get_momentum_features(df):
    lep = _calculate_momenta(df, 'PRI_lep_')
    jet_leading = _calculate_momenta(df, 'PRI_jet_leading_')
    jet_subleading = _calculate_momenta(df, 'PRI_jet_subleading_')
    tau = _calculate_momenta(df, 'PRI_tau_')
    return lep.join(tau).join(jet_leading).join(jet_subleading)

def get_abs_momentum_features(df):
    lep = _get_abs_momenta(df, 'PRI_lep_')
    jet_leading = _get_abs_momenta(df, 'PRI_jet_leading_')
    jet_subleading = _get_abs_momenta(df, 'PRI_jet_subleading_')
    tau = _get_abs_momenta(df, 'PRI_tau_')
    return lep.join(tau).join(jet_leading).join(jet_subleading)

def with_momentum_features(df):
    return df.join(get_momentum_features(df)).replace([np.inf, -np.inf], np.nan).fillna(-999.)

def with_abs_momentum_features(df):
    return df.join(get_abs_momentum_features(df)).replace([np.inf, -np.inf], np.nan).fillna(-999.)

# sum PT feature of vector summations

def pt_sqrt(x,y):
    return np.sqrt(x*x+y*y)

def tau_lep_vec_sum_pt(row):
    x = row['PRI_tau_px']+row['PRI_lep_px']
    y = row['PRI_tau_py']+row['PRI_lep_py']
    frame = pt_sqrt(x,y)
    frame.name = 'New_tau_lep_pt_vec_sum'
    return frame

def tau_jet_vec_sum_pt(row):
    x = row['PRI_tau_px']+row['PRI_jet_leading_px']
    y = row['PRI_tau_py']+row['PRI_jet_leading_py']
    frame = pt_sqrt(x,y)
    frame.name = 'New_tau_jet_pt_vec_sum'
    mask = (row['PRI_jet_num']==0)
    frame[mask]=-999.
    return frame
    
def lep_jet_vec_sum_pt(row):
    x = row['PRI_lep_px']+row['PRI_jet_leading_px']
    y = row['PRI_lep_py']+row['PRI_jet_leading_py']
    frame = pt_sqrt(x,y)
    frame.name = 'New_lep_jet_pt_vec_sum'
    mask = (row['PRI_jet_num']==0)
    frame[mask]=-999.
    return frame

vec_pt_features = [tau_lep_vec_sum_pt,tau_jet_vec_sum_pt,lep_jet_vec_sum_pt]
# Eta Features

# The distance from the x=y line in jet eta0, eta1 space
def eta_plus(x, y):
    return np.sqrt(x*x/2 + 2*y*y - 2*x*y)

def jet_eta_plus(row):
    x = row['PRI_jet_leading_eta']
    y = row['PRI_jet_subleading_eta']
    frame = eta_plus(x, y)
    frame.name = 'New_jet_eta_plus'
    #mask = (row['PRI_jet_subleading_eta']<-10.)
    mask = (row['PRI_jet_num']<2)
    frame[mask]==-999.0
    return frame #eta_plus(x, y)

# Do the same with the lepton and tau
def lep_tau_eta_plus(row):
    x = row['PRI_lep_eta']
    y = row['PRI_tau_eta']
    frame = eta_plus(x, y)
    frame.name = 'New_lep_tau_eta_plus'
    return frame #eta_plus(x, y)

# Do the same with the lepton and jet
def lep_jet_eta_plus(row):
    x = row['PRI_lep_eta']
    y = row['PRI_jet_leading_eta']
    frame = eta_plus(x, y)
    frame.name = 'New_lep_jet_eta_plus'
    return frame #eta_plus(x, y)

# Do the same with the tau and jet
def tau_jet_eta_plus(row):
    x = row['PRI_tau_eta']
    y = row['PRI_jet_leading_eta']
    frame = eta_plus(x, y)
    frame.name = 'New_tau_jet_eta_plus'
    return frame #eta_plus(x, y)

def jet_leading_abs_eta(row):
    frame = row['PRI_jet_leading_eta'].map(lambda x:abs(x))
    frame.name = 'PRI_jet_leading_eta_abs'
    mask = (row['PRI_jet_num']==0)
    frame[mask]=-999.
    return frame

def jet_subleading_abs_eta(row):
    frame = row['PRI_jet_subleading_eta'].map(lambda x:abs(x))
    frame.name = 'PRI_jet_subleading_eta_abs'
    mask = (row['PRI_jet_num']<2)
    frame[mask]=-999.
    return frame

rapidity_features = [jet_eta_plus, lep_jet_eta_plus, tau_jet_eta_plus,
                     lep_tau_eta_plus,
                     jet_leading_abs_eta,
                     jet_subleading_abs_eta
                     ]


# Z Momentum Features: only when jet is valid

def lep_z_diff_momemtum(row):
    frame = (row['PRI_lep_pz'] - row['PRI_tau_pz']).map(lambda x: abs(x))#make it non symmetric
    frame.name = 'New_lep_z_diff_momentum'
    return frame

def lep_z_momentum(row):
    frame = (row['PRI_lep_pz'] + row['PRI_tau_pz']).map(lambda x: abs(x))
    frame.name = 'New_lep_z_momentum'
    return frame

def jet_z_momentum(row):
    frame = row['PRI_jet_leading_pz'] + row['PRI_jet_subleading_pz']
    frame.name = 'New_jet_z_momentum'
    #mask = (row['PRI_jet_subleading_pt']==-999.) #two valid jets can have j1z-j2z
    #mask = (row['PRI_jet_subleading_pt']<0.0)
    mask = (row['PRI_jet_num']<2)
    frame[mask]==-999.
    return frame

def jet_lep_sum_z_momentum(row):
    frame = lep_z_momentum(row) + jet_z_momentum(row)
    frame.name = 'New_jet_lep_sum_z_momentum'
    mask = (row['PRI_jet_num']<2)
    frame[mask]==-999.
    return frame

def jet_lep_diff_z_momentum(row):
    frame = lep_z_momentum(row) - jet_z_momentum(row)
    frame.name = 'New_jet_lep_diff_z_momentum'
    mask = (row['PRI_jet_num']<2)
    frame[mask]==-999.
    return frame

#09-11-2014 add lep_z_diff_momemtum
z_momentum_features = [lep_z_momentum, lep_z_diff_momemtum,
                       #09-11-2014 try removing these confusing jet-z momentums
                       jet_z_momentum, jet_lep_sum_z_momentum, jet_lep_diff_z_momentum
                       ]


# Transverse Momenta Features

def max_jet_pt(row):
    #frame = max(row['PRI_jet_leading_pt'], row['PRI_jet_subleading_pt'])  
    frame = row[['PRI_jet_leading_pt','PRI_jet_subleading_pt']].max(axis=1)  
    frame.name = 'New_max_jet_pt'
    return frame

def min_jet_pt(row):
    #frame = min(row['PRI_jet_leading_pt'], row['PRI_jet_subleading_pt'])  
    frame = row[['PRI_jet_leading_pt','PRI_jet_subleading_pt']].min(axis=1)
    frame.name = 'New_min_jet_pt'
    return frame

def max_lep_pt(row):
    frame = row[['PRI_tau_pt','PRI_lep_pt']].max(axis=1) 
    frame.name = 'New_max_lep_pt'
    return frame

def min_lep_pt(row):
    frame = row[['PRI_tau_pt','PRI_lep_pt']].min(axis=1) 
    frame.name = 'New_min_lep_pt'
    return frame

def max_pt(row):
    #frame = max_jet_pt(row).join(max_lep_pt(row))
    max_jet_pt_frame = pandas.DataFrame(max_jet_pt(row),columns=['max_jet_pt'])
    max_lep_pt_frame = pandas.DataFrame(max_lep_pt(row),columns=['max_lep_pt'])
    frame = max_jet_pt_frame.join(max_lep_pt_frame).max(axis=1)
    frame.name = 'New_max_pt'
    return frame

def min_pt(row):
    min_jet_pt_frame = pandas.DataFrame(min_jet_pt(row),columns=['min_jet_pt'])
    min_lep_pt_frame = pandas.DataFrame(min_lep_pt(row),columns=['min_lep_pt'])
    frame = min_jet_pt_frame.join(min_lep_pt_frame).min(axis=1)
    frame.name = 'New_min_pt'
    return frame

def sum_jet_pt(row):#??
    frame = row['PRI_jet_leading_pt'] + row['PRI_jet_subleading_pt']
    frame.name = 'New_sum_jet_pt'
    mask = (row['PRI_jet_subleading_phi']==-999.0)
    frame[mask]=-999.0
    return frame

def sum_jet_vec_pt(row):#??
    x = row['PRI_jet_leading_px'] + row['PRI_jet_subleading_px']
    y = row['PRI_jet_leading_py'] + row['PRI_jet_subleading_py']
    frame = pt_sqrt(x,y)
    frame.name = 'New_sum_jet_vec_pt'
    mask = (row['PRI_jet_num']<2)
    frame[mask]=-999.0
    return frame

def sum_lep_pt(row):#should be replaced by the tau-lep vect sum pt
    frame = row['PRI_tau_pt'] + row['PRI_lep_pt']
    frame.name = 'New_sum_lep_pt'
    return frame

#pt(ditau) ~ pt_tau+pt_p_j+met for one jet > 140GeV
def pt_tautau_single_jet(row):
    frame = row['PRI_tau_pt']+row['PRI_jet_leading_pt']+row['PRI_met']
    frame.name = 'New_pt_tautau_single_jet'
    #mask = (row['PRI_jet_leading_pt']==-999.0)
    #mask = (row['PRI_jet_subleading_pt']==-999.0)
    mask = (row['PRI_jet_num']==0)
    frame[mask]==-999.0
    return frame

#pt(ditau) ~ pt_tau+pt_p_j+met for two jet >110 GeV with delta eta jj > 2.5
def pt_tautau_multi_jet(row):
    frame = row['PRI_tau_pt']+row['PRI_jet_leading_pt']+row['PRI_jet_subleading_pt']+row['PRI_met']
    frame.name = 'New_pt_tautau_multi_jet'
    mask = (row['PRI_jet_num']<2)
    frame[mask]==-999.0
    return frame

transverse_momentum_features = [max_jet_pt, min_jet_pt, max_lep_pt, min_lep_pt,
                                #max_pt, min_pt, 
                                #09-11-2014 remove these confusing and mis-treat of missing value
                                #pt_tautau_single_jet, pt_tautau_multi_jet,
                                sum_jet_pt, sum_lep_pt
                                ]


# Momentum Ratio Features

def frac_tau_pt(row):
    tau_pt = row['PRI_tau_pt']
    lep_pt = row['PRI_lep_pt']
    frame = tau_pt / (tau_pt + lep_pt)
    frame.name = 'New_frac_tau_pt'
    return frame

def frac_lep_pt(row):
    tau_pt = row['PRI_tau_pt']
    lep_pt = row['PRI_lep_pt']
    frame = lep_pt / (tau_pt + lep_pt)
    frame.name = 'New_frac_lep_pt'
    return frame

def frac_tau_p(row):
    tau_p = row['PRI_tau_p_tot']
    lep_p = row['PRI_lep_p_tot']
    frame = tau_p / (tau_p + lep_p)
    frame.name = 'New_frac_tau_p'
    return frame

def frac_lep_p(row):
    tau_p = row['PRI_tau_p_tot']
    lep_p = row['PRI_lep_p_tot']
    frame = lep_p / (tau_p + lep_p)
    frame.name = 'New_frac_lep_p'
    return frame

momentum_ratio_features = [frac_tau_pt, frac_lep_pt, frac_tau_p, frac_lep_p]


# MET Features

def ht(row):
    frame = sum_jet_pt(row) + sum_lep_pt(row)
    frame.name = 'New_ht'
    mask = (row['PRI_jet_subleading_phi']==-999.0)
    frame[mask]=-999.0
    return frame

def ht_met(row):
    frame = ht(row) + row['PRI_met']
    frame.name = 'New_ht_met'
    mask = (row['PRI_jet_subleading_phi']==-999.0)
    frame[mask]=-999.0
    return frame

#met tau phi have no -999 values, so they are fine
def tau_met_cos_phi(row):
    #frame = math.cos(row['PRI_met_phi'] - row['PRI_tau_phi'])
    frame = (row['PRI_met_phi'] - row['PRI_tau_phi']).map(lambda x: math.cos(x))
    frame.name = 'New_tau_met_cos_phi'
    return frame

def lep_met_cos_phi(row):
    frame = (row['PRI_met_phi'] - row['PRI_lep_phi']).map(lambda x: math.cos(x))
    frame.name = 'New_lep_met_cos_phi'
    return frame

#jet phi can have -999.0 value so direct calculation may cause confusion to GBM
def jet_leading_met_cos_phi(row):
    frame = (row['PRI_jet_leading_phi'] - row['PRI_lep_phi']).map(lambda x: math.cos(x))
    frame.name = 'New_jet_leading_met_cos_phi'
    mask = (row['PRI_jet_leading_phi']==-999.0)
    frame[mask]=-999.0
    return frame

def jet_subleading_met_cos_phi(row):
    frame = (row['PRI_jet_subleading_phi'] - row['PRI_lep_phi']).map(lambda x: math.cos(x))
    frame.name = 'New_jet_subleading_met_cos_phi'
    mask = (row['PRI_jet_subleading_phi']==-999.0)
    frame[mask]=-999.0
    return frame

def tau_met_sin_phi(row):
    frame = (row['PRI_met_phi'] - row['PRI_tau_phi']).map(lambda x: math.sin(x))
    frame.name = 'New_tau_met_sin_phi'
    return frame

def lep_met_sin_phi(row):
    #frame = (row['PRI_met_phi'] - row['PRI_lep_phi']).map(lambda x: math.sin(x))
    #09-11-2014 sin is symmetic for it, make it abs
    frame = (row['PRI_met_phi'] - row['PRI_lep_phi']).map(lambda x: abs(math.sin(x)))
    frame.name = 'New_lep_met_sin_phi'
    return frame

def lep_met_delta_phi(row):#just open angle combinations: 09-11 make it non-symmatric so a cut is valid
    #make it cycle as pi
    frame = (row['PRI_lep_phi'] - row['PRI_met_phi']).map(lambda x: abs(x%(2*math.pi)-math.pi))
    frame.name = 'New_lep_met_deltaphi'
    return frame

def tau_met_delta_phi(row):#just open angle combinations: 09-11: still no separation after pi/2 normalization
    #frame = (row['PRI_tau_phi'] - row['PRI_met_phi']).map(lambda x: abs(x%(2*math.pi)-math.pi))
    #make it cycle as pi/2
    frame = (row['PRI_tau_phi'] - row['PRI_met_phi']).map(lambda x: abs(abs(x%(2*math.pi)-math.pi)-math.pi/2))
    frame.name = 'New_tau_met_deltaphi'
    return frame

def tau_lep_delta_phi(row):#just open angle combinations: small angle separation
    frame = (row['PRI_tau_phi'] - row['PRI_lep_phi']).map(lambda x: abs(x%(2*math.pi)-math.pi))
    frame.name = 'New_tau_lep_deltaphi'
    return frame

def jet_leading_met_sin_phi(row):
    frame = (row['PRI_jet_leading_phi'] - row['PRI_lep_phi']).map(lambda x: math.sin(x))
    frame.name = 'New_jet_leading_met_sin_phi'
    mask = (row['PRI_jet_leading_phi']==-999.0)
    frame[mask]=-999.0
    return frame

def jet_subleading_met_sin_phi(row):
    frame = (row['PRI_jet_subleading_phi'] - row['PRI_lep_phi']).map(lambda x: math.sin(x))
    frame.name = 'New_jet_subleading_met_sin_phi'
    mask = (row['PRI_jet_subleading_phi']==-999.0)
    frame[mask]=-999.0
    return frame

def jet_leading_tau_cos_phi(row):
    frame = (row['PRI_jet_leading_phi'] - row['PRI_tau_phi']).map(lambda x: math.cos(x))
    frame.name = 'New_jet_leading_tau_cos_phi'
    mask = (row['PRI_jet_leading_phi']==-999.0)
    frame[mask]=-999.0
    return frame

def jet_leading_tau_sin_phi(row):
    frame = (row['PRI_jet_leading_phi'] - row['PRI_tau_phi']).map(lambda x: math.sin(x))
    frame.name = 'New_jet_leading_tau_sin_phi'
    mask = (row['PRI_jet_leading_phi']==-999.0)
    frame[mask]=-999.0
    return frame

#w+jet invariant mass : sqrt(2*pt*MET*(1-cos phi))
def w_jet_invariant_mass(row):
    frame = np.sqrt(tau_met_cos_phi(row).map(lambda x:(1-x)*2)*row['PRI_met']*row['PRI_jet_leading_pt'])
    frame = frame.map(lambda x: x if not math.isnan(x) else -999.)
    frame.name = 'New_W_jet_invariant_mass'
    mask = (row['PRI_jet_leading_pt']==-999.0)
    frame[mask]=-999.0
    return frame

#Z->ll in the background!
def tau_lepton_invariant_mass(row):
    frame = np.sqrt(lepton_tau_cos_phi(row).map(lambda x:(1-x)*2)*row['PRI_lep_pt']*row['PRI_tau_pt'])
    frame = frame.map(lambda x: x if not math.isnan(x) else -999.)
    frame.name = 'New_tau_lep_invariant_mass'
    return frame

def met_jet_invariant_mass(row):
    frame = np.sqrt(jet_leading_met_cos_phi(row).map(lambda x:(1-x)*2)*row['PRI_met']*row['PRI_jet_leading_pt'])
    frame = frame.map(lambda x: x if not math.isnan(x) else -999.)
    frame.name = 'New_met_jet_invariant_mass'
    mask = (row['PRI_jet_leading_pt']==-999.0)
    frame[mask]=-999.0
    return frame

def tau_jet_invariant_mass(row):
    frame = np.sqrt(jet_leading_tau_cos_phi(row).map(lambda x:(1-x)*2)*row['PRI_tau_pt']*row['PRI_jet_leading_pt'])
    frame = frame.map(lambda x: x if not math.isnan(x) else -999.)
    frame.name = 'New_tau_jet_invariant_mass'
    mask = (row['PRI_jet_leading_pt']==-999.0)
    frame[mask]=-999.0
    return frame


def min_met_cos_phi(row):
    #frame = tau_met_cos_phi(row).join(lep_met_cos_phi(row)).join(jet_leading_met_cos_phi(row)).join(jet_subleading_met_cos_phi(row)).min(axis=1)
    tau_met_cos_phi_frame= pandas.DataFrame(tau_met_cos_phi(row),columns=['tau_met_cos_phi'])
    lep_met_cos_phi_frame= pandas.DataFrame((row),columns=['lep_met_cos_phi'])
    frame = tau_met_cos_phi_frame.join(lep_met_cos_phi_frame).min(axis=1)
    frame.name = 'New_min_met_cos_phi'
    return frame

def max_met_cos_phi(row):
    #frame = tau_met_cos_phi(row).join(lep_met_cos_phi(row)).join(jet_leading_met_cos_phi(row)).join(jet_subleading_met_cos_phi(row)).max(axis=1)
    tau_met_cos_phi_frame= pandas.DataFrame(tau_met_cos_phi(row),columns=['tau_met_cos_phi'])
    lep_met_cos_phi_frame= pandas.DataFrame((row),columns=['lep_met_cos_phi'])
    frame = tau_met_cos_phi_frame.join(lep_met_cos_phi_frame).max(axis=1)
    frame.name = 'New_max_met_cos_phi'
    return frame

def met_sig(row):
    frame = row['PRI_met'] / np.sqrt(row['PRI_met_sumet'])
    frame = frame.map(lambda x: x if not math.isnan(x) else -999.)
    frame.name = 'New_met_sig'
    return frame

def sumet_sum_pt_ratio(row):
    frame = row['PRI_met_sumet'] / row['DER_sum_pt']
    frame = frame.map(lambda x: x if not math.isnan(x) else -999.)
    frame.name = 'New_sumet_sum_pt_ratio'
    return frame

def sumet_tau_lep_sum_pt_ratio(row):
    frame = row['PRI_met_sumet'] / tau_lep_vec_sum_pt(row)
    frame.name = 'New_sumet_tau_lep_sum_pt_ratio'
    return frame

def sumet_tau_jet_sum_pt_ratio(row):
    frame = row['PRI_met_sumet'] / tau_jet_vec_sum_pt(row)
    frame.name = 'New_sumet_tau_jet_sum_pt_ratio'
    mask = (row['PRI_jet_num']==0)
    frame[mask]==-999.
    return frame

def sumet_lep_jet_sum_pt_ratio(row):
    frame = row['PRI_met_sumet'] / lep_jet_vec_sum_pt(row)
    frame.name = 'New_sumet_lep_jet_sum_pt_ratio'
    mask = (row['PRI_jet_num']==0)
    frame[mask]==-999.
    return frame

def met_pt_total_ratio(row):
    #if (row['DER_pt_tot']==0):
    #    frame = row['DER_pt_tot']
    #else:
    frame =  row['PRI_met'] / row['DER_pt_tot']
    #frame = frame.map(lambda x: x if not math.isnan(x) else -999.)
    frame.name = 'New_met_pt_total_ratio'
    mask = (row['DER_pt_tot']==0)
    frame[mask]==-999.0
    return frame

met_features = [ht, ht_met, 
                #09-11-2014 remove tau_met_lep sin cos and put the raw open angle to see if it 
                #still hurt the leaderboard
                #tau_met_cos_phi, 
                lep_met_cos_phi, #jet_leading_met_cos_phi, jet_subleading_met_cos_phi,
                #tau_met_sin_phi, #turns out tau met has no much separation
                lep_met_sin_phi, #jet_leading_met_sin_phi, jet_subleading_met_sin_phi,
                #jet_leading_tau_cos_phi, jet_leading_tau_sin_phi,
                #open angles have separation, but it hurts the public board.....don't know why
                #give another try of open angle absolute value of providing a cut
                tau_lep_delta_phi, 
                #tau_met_delta_phi, #tested...no separation
                lep_met_delta_phi, #just open angle combinations
                min_met_cos_phi, max_met_cos_phi, 
                #met-jet, tau-jet, tau-lep invariant mass have no much sense but helps
                w_jet_invariant_mass, met_jet_invariant_mass, tau_jet_invariant_mass,tau_lepton_invariant_mass,
                met_sig, 
                #09-11-2014 add sumet/tau-lep-vec sumet/tau-jet-vect sumet/tau-lep-vect
                sumet_sum_pt_ratio, sumet_tau_lep_sum_pt_ratio, 
                #sumet_tau_jet_sum_pt_ratio, #almost no separation, no add
                #sumet_lep_jet_sum_pt_ratio, #almost no separation, no add
                #met_pt_total_ratio #sorry remove, almost no separation
                ]


# Jet Features

def jet_delta_cos_phi(row):
    frame = (row['PRI_jet_leading_phi'] - row['PRI_jet_subleading_phi']).map(lambda x: math.cos(x))
    mask = (row['PRI_jet_subleading_phi']==-999.0)
    frame.name = 'New_jet_delta_cos_phi'
    frame[mask]=-999.
    return frame

#delta eta is large for jet-jet, see CMS tau tau
def jet_delta_eta(row):
    frame = (row['PRI_jet_leading_eta']-row['PRI_jet_subleading_eta']).map(lambda x: abs(x))
    mask = (row['PRI_jet_subleading_eta']==-999.0)
    frame.name = 'New_jet_delta_eta'
    frame[mask]=-999.
    return frame

def jet_tau_delta_eta(row):
    frame = (row['PRI_jet_leading_eta']-row['PRI_tau_eta']).map(lambda x: abs(x))
    mask = (row['PRI_jet_leading_eta']==-999.0)
    frame.name = 'New_jet_tau_delta_eta'
    frame[mask]=-999.
    return frame

def lep_tau_delta_eta(row):
    frame = (row['PRI_lep_eta']-row['PRI_tau_eta']).map(lambda x: abs(x))
    frame.name = 'New_lep_tau_delta_eta'
    return frame

def jet_lep_delta_eta(row):
    frame = (row['PRI_jet_leading_eta']-row['PRI_lep_eta']).map(lambda x: abs(x))
    mask = (row['PRI_jet_leading_eta']==-999.0)
    frame.name = 'New_jet_lep_delta_eta'
    frame[mask]=-999.
    return frame

#sqrt(2*jet_pt1*jet_pt2*(1-cos phi)) when dijets
def jet_jet_invariant_mass(row):
    frame = np.sqrt(jet_delta_cos_phi(row).map(lambda x:(1-x)*2)*row['PRI_jet_subleading_pt']*row['PRI_jet_leading_pt'])
    frame = frame.map(lambda x: x if not math.isnan(x) else -999.)
    frame.name = 'New_jet_jet_invariant_mass'
    mask = (row['PRI_jet_subleading_pt']==-999.0)
    frame[mask]=-999.
    return frame

jet_features = [jet_delta_cos_phi,
                #because of the jet nature, its eta open angle with other ones are important
                jet_delta_eta, jet_tau_delta_eta, lep_tau_delta_eta, jet_lep_delta_eta,
                jet_jet_invariant_mass]

def lepton_tau_cos_phi(row):
    frame = (row['PRI_lep_phi']-row['PRI_tau_phi']).map(lambda x: math.cos(x))
    frame = frame.map(lambda x: x if not math.isnan(x) else -999.)
    frame.name = 'New_lep_tau_cos_phi'
    return frame

#MOVE to invariant mass part
def lepton_tau_invariant_mass(row):
    frame = np.sqrt(lepton_tau_cos_phi(row).map(lambda x:(1-x)*2)*row['PRI_lep_pt']*row['PRI_tau_pt'])
    frame = frame.map(lambda x: x if not math.isnan(x) else -999.)
    frame.name = 'New_lepton_tau_invariant_mass'
    return frame

lepton_tau_features = [lepton_tau_cos_phi, 
                       #lepton_tau_invariant_mass
                       ]

#for centrality sign calculation
def tau_lep_sign_sin_phi(row):
    frame = (row['PRI_tau_phi']-row['PRI_lep_phi']).map(lambda x: math.sin(x))
    #frame = (row['PRI_tau_phi']-row['PRI_met_phi']).map(lambda x: math.sin(x))
    frame = frame.map(lambda x: x if not math.isnan(x) else -999.)
    frame.name = 'New_tau_lep_sign_sin_phi'
    return frame

#positive negative categorial feat of tau met
def tau_met_lep_sin_phi_sign(row):
    #frame = tau_met_sin_phi(row).map(lambda x: 1 if x>0. else 0)
    frame = tau_lep_sign_sin_phi(row).map(lambda x: 1 if x>0. else 0)
    frame.name = 'New_tau_met_lep_sin_phi_sign'
    return frame

def signed_met_phi_centrality_w_lep(row):#NOPE the original one was correct, no need sign correction
    #original DER_met_phi_centrality has only absolute value with no sign
    frame = row['DER_met_phi_centrality']*tau_met_lep_sin_phi_sign(row)
    frame.name = 'New_DER_met_phi_centrality_w_tau_plus_lep'
    return frame

#positive negative categorial feat of tau met
def tau_met_sin_phi_sign(row):
    #frame = tau_met_sin_phi(row).map(lambda x: 1 if x>0. else 0)
    frame = tau_met_sin_phi(row).map(lambda x: 1 if x>0. else 0)
    frame.name = 'New_tau_met_sin_phi_sign'
    return frame

def signed_met_phi_centrality(row):#NOPE the original one was correct, no need sign correction
    #original DER_met_phi_centrality has only absolute value with no sign
    frame = row['DER_met_phi_centrality']*tau_met_sin_phi_sign(row)
    frame.name = 'New_DER_met_phi_centrality_w_tau'
    return frame

def log_lep_eta_centrality(row):
    #lep eta centrality is exp(-4*....), so a log feat may help
    frame = row['DER_lep_eta_centrality'].map(lambda x: -999. if x<0.01 else math.log(x))
    frame.name = 'New_log_lep_eta_centrality'
    return frame

# Adding features to a DF

centrality_features = [#tau_met_sin_phi, 
                       #tau_met_sin_phi_sign
                       #signed_met_phi_centrality, #09-11 add the raw signed centrality of tau-met (no use .. )
                       signed_met_phi_centrality_w_lep,
                       log_lep_eta_centrality
                       ]


def add_features(df):

    new_features = []
    new_features.extend(rapidity_features)
    new_features.extend(z_momentum_features)
    new_features.extend(transverse_momentum_features)
    new_features.extend(momentum_ratio_features)
    new_features.extend(met_features)
    new_features.extend(jet_features)
    new_features.extend(lepton_tau_features)
    new_features.extend(centrality_features) #expand the centrality with sin(tau-lep) and log(cent(tau-lep))
    new_features.extend(vec_pt_features)
    #categorizing jet is a bad idea, it drops from 3.75 to 3.68
    #new_features.extend(jet_num_feat)

    def with_new_features(df):
        #return df.join(map_functions(df, new_features))
        for f in new_features:
            df = df.join(f(df))
        return df

    df_all_features = with_momentum_features(df)
    df_all_features = with_abs_momentum_features(df_all_features)
    df_all_features = with_new_features(df_all_features)

    return df_all_features
