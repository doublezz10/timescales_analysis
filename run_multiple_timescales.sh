#!/bin/sh

cd /Users/zachz/Documents/timescales_analysis/Buzsaki

Python acc_make_fake_trials.py &
Python bla_make_fake_trials.py &
python centralNuc_make_fake_trials.py &
Python hippocampus_make_fake_trials.py &
Python mpfc_make_fake_trials.py &
Python ofc_make_fake_trials.py

cd /Users/zachz/Documents/timescales_analysis/Faraut

Python faraut_amygdala_all_plots.py &
Python faraut_hippocampus_all_plots.py

cd /Users/zachz/Documents/timescales_analysis/FROOT

python froot_analysis.py

cd /Users/zachz/Documents/timescales_analysis/LeMerre

python make_fake_trials_lag_loop.py

cd /Users/zachz/Documents/timescales_analysis/Meg

python meg_amygdala_all_plots.py &
python meg_scACC_all_plots.py &
python meg_ventralStriatum_all_plots.py

cd /Users/zachz/Documents/timescales_analysis/Minxha

python minxha_amygdala_all_plots.py &
python minxha_dacc_all_plots.py &
python minxha_hippocampus_all_plots.py &
python minxha_presma_all_plots.py

cd /Users/zachz/Documents/timescales_analysis/Peyrache

python make_fake_trials.py

cd /Users/zachz/Documents/timescales_analysis/Steinmetz

python steinmetz_aca_all_plots.py &
python steinmetz_bla_all_plots.py &
python steinmetz_ca1_all_plots.py &
python steinmetz_ca2_all_plots.py &
python steinmetz_ca3_all_plots.py &
python steinmetz_dg_all_plots.py &
python steinmetz_ila_all_plots.py &
python steinmetz_orb_all_plots.py &
python steinmetz_pl_all_plots.py

cd /Users/zachz/Documents/timescales_analysis/Stoll

python stoll_amyg_all_plots.py &
python stoll_cd_all_plots.py &
python stoll_dlpfc_all_plots.py &
python stoll_ifg_all_plots.py &
python stoll_lai_all_plots.py &
python stoll_ofc_all_plots.py &
python stoll_PMd_all_plots.py &
python stoll_put_all_plots.py &
python stoll_vlpfc_all_plots.py

cd /Users/zachz/Documents/timescales_analysis/Wirth

python make_fake_trials_1.py
python make_fake_trials_2.py
