%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%                                                                  %%%%%
%%%%    IEEE PES Power Grid Library - Optimal Power Flow - v19.05     %%%%%
%%%%          (https://github.com/power-grid-lib/pglib-opf)           %%%%%
%%%%             Benchmark Group - Active Power Increase              %%%%%
%%%%                         10 - May - 2019                          %%%%%
%%%%                                                                  %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function mpc = pglib_opf_case30_fsr__api
mpc.version = '2';
mpc.baseMVA = 100.0;

%% area data
%	area	refbus
mpc.areas = [
	1	 8;
	2	 23;
	3	 26;
];

%% bus data
%	bus_i	type	Pd	Qd	Gs	Bs	area	Vm	Va	baseKV	zone	Vmax	Vmin
mpc.bus = [
	1	 3	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 135.0	 1	    1.05000	    0.95000;
	2	 2	 23.71	 12.70	 0.0	 0.0	 1	    1.02500	    0.00000	 135.0	 1	    1.10000	    0.95000;
	3	 1	 2.62	 1.20	 0.0	 0.0	 1	    1.00000	    0.00000	 135.0	 1	    1.05000	    0.95000;
	4	 1	 8.30	 1.60	 0.0	 0.0	 1	    1.00000	    0.00000	 135.0	 1	    1.05000	    0.95000;
	5	 1	 0.0	 0.0	 0.0	 0.19	 1	    1.00000	    0.00000	 135.0	 1	    1.05000	    0.95000;
	6	 1	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 135.0	 1	    1.05000	    0.95000;
	7	 1	 24.91	 10.90	 0.0	 0.0	 1	    1.00000	    0.00000	 135.0	 1	    1.05000	    0.95000;
	8	 1	 32.78	 30.00	 0.0	 0.0	 1	    1.00000	    0.00000	 135.0	 1	    1.05000	    0.95000;
	9	 1	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 135.0	 1	    1.05000	    0.95000;
	10	 1	 6.34	 2.00	 0.0	 0.0	 3	    1.00000	    0.00000	 135.0	 1	    1.05000	    0.95000;
	11	 1	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 135.0	 1	    1.05000	    0.95000;
	12	 1	 12.24	 7.50	 0.0	 0.0	 2	    1.00000	    0.00000	 135.0	 1	    1.05000	    0.95000;
	13	 2	 0.0	 0.0	 0.0	 0.0	 2	    1.02500	    0.00000	 135.0	 1	    1.10000	    0.95000;
	14	 1	 6.77	 1.60	 0.0	 0.0	 2	    1.00000	    0.00000	 135.0	 1	    1.05000	    0.95000;
	15	 1	 8.96	 2.50	 0.0	 0.0	 2	    1.00000	    0.00000	 135.0	 1	    1.05000	    0.95000;
	16	 1	 3.82	 1.80	 0.0	 0.0	 2	    1.00000	    0.00000	 135.0	 1	    1.05000	    0.95000;
	17	 1	 9.83	 5.80	 0.0	 0.0	 2	    1.00000	    0.00000	 135.0	 1	    1.05000	    0.95000;
	18	 1	 3.50	 0.90	 0.0	 0.0	 2	    1.00000	    0.00000	 135.0	 1	    1.05000	    0.95000;
	19	 1	 10.38	 3.40	 0.0	 0.0	 2	    1.00000	    0.00000	 135.0	 1	    1.05000	    0.95000;
	20	 1	 2.40	 0.70	 0.0	 0.0	 2	    1.00000	    0.00000	 135.0	 1	    1.05000	    0.95000;
	21	 1	 19.12	 11.20	 0.0	 0.0	 3	    1.00000	    0.00000	 135.0	 1	    1.05000	    0.95000;
	22	 2	 0.0	 0.0	 0.0	 0.0	 3	    1.02500	    0.00000	 135.0	 1	    1.10000	    0.95000;
	23	 2	 3.50	 1.60	 0.0	 0.0	 2	    1.02500	    0.00000	 135.0	 1	    1.10000	    0.95000;
	24	 1	 9.51	 6.70	 0.0	 0.04	 3	    1.00000	    0.00000	 135.0	 1	    1.05000	    0.95000;
	25	 1	 0.0	 0.0	 0.0	 0.0	 3	    1.00000	    0.00000	 135.0	 1	    1.05000	    0.95000;
	26	 1	 3.82	 2.30	 0.0	 0.0	 3	    1.00000	    0.00000	 135.0	 1	    1.05000	    0.95000;
	27	 2	 0.0	 0.0	 0.0	 0.0	 3	    1.02500	    0.00000	 135.0	 1	    1.10000	    0.95000;
	28	 1	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 135.0	 1	    1.05000	    0.95000;
	29	 1	 2.62	 0.90	 0.0	 0.0	 3	    1.00000	    0.00000	 135.0	 1	    1.05000	    0.95000;
	30	 1	 11.58	 1.90	 0.0	 0.0	 3	    1.00000	    0.00000	 135.0	 1	    1.05000	    0.95000;
];

%% generator data
%	bus	Pg	Qg	Qmax	Qmin	Vg	mBase	status	Pmax	Pmin
mpc.gen = [
	1	 133.5	 39.8	 250.0	 -170.4	 1.0	 100.0	 1	 267	 0.0; % NG
	2	 9.0	 0.0	 159.6	 -159.6	 1.025	 100.0	 1	 18	 0.0; % NG
	22	 53.0	 13.5	 80.0	 -53.0	 1.025	 100.0	 1	 106	 0.0; % NG
	27	 109.0	 0.0	 109.0	 -109.0	 1.025	 100.0	 1	 218	 0.0; % NG
	23	 68.5	 0.0	 69.0	 -69.0	 1.025	 100.0	 1	 137	 0.0; % NG
	13	 51.5	 4.0	 60.0	 -52.0	 1.025	 100.0	 1	 103	 0.0; % NG
];

%% generator cost data
%	2	startup	shutdown	n	c(n-1)	...	c0
mpc.gencost = [
	2	 0.0	 0.0	 3	   0.020000	   2.000000	   0.000000; % NG
	2	 0.0	 0.0	 3	   0.017500	   1.750000	   0.000000; % NG
	2	 0.0	 0.0	 3	   0.062500	   1.000000	   0.000000; % NG
	2	 0.0	 0.0	 3	   0.008340	   3.250000	   0.000000; % NG
	2	 0.0	 0.0	 3	   0.025000	   3.000000	   0.000000; % NG
	2	 0.0	 0.0	 3	   0.025000	   3.000000	   0.000000; % NG
];

%% branch data
%	fbus	tbus	r	x	b	rateA	rateB	rateC	ratio	angle	status	angmin	angmax
mpc.branch = [
	1	 2	 0.0192	 0.0575	 0.0264	 130.0	 130.0	 130.0	 0.0	 0.0	 1	 -30.0	 30.0;
	1	 3	 0.0452	 0.1852	 0.0204	 130.0	 130.0	 130.0	 0.0	 0.0	 1	 -30.0	 30.0;
	2	 4	 0.057	 0.1737	 0.0184	 65.0	 65.0	 65.0	 0.0	 0.0	 1	 -30.0	 30.0;
	3	 4	 0.0132	 0.0379	 0.0042	 130.0	 130.0	 130.0	 0.0	 0.0	 1	 -30.0	 30.0;
	2	 5	 0.0472	 0.1983	 0.0209	 130.0	 130.0	 130.0	 0.0	 0.0	 1	 -30.0	 30.0;
	2	 6	 0.0581	 0.1763	 0.0187	 65.0	 65.0	 65.0	 0.0	 0.0	 1	 -30.0	 30.0;
	4	 6	 0.0119	 0.0414	 0.0045	 90.0	 90.0	 90.0	 0.0	 0.0	 1	 -30.0	 30.0;
	5	 7	 0.046	 0.116	 0.0102	 70.0	 70.0	 70.0	 0.0	 0.0	 1	 -30.0	 30.0;
	6	 7	 0.0267	 0.082	 0.0085	 130.0	 130.0	 130.0	 0.0	 0.0	 1	 -30.0	 30.0;
	6	 8	 0.012	 0.042	 0.0045	 32.0	 32.0	 32.0	 0.0	 0.0	 1	 -30.0	 30.0;
	6	 9	 0.0	 0.208	 0.0	 65.0	 65.0	 65.0	 0.0	 0.0	 1	 -30.0	 30.0;
	6	 10	 0.0	 0.556	 0.0	 32.0	 32.0	 32.0	 0.0	 0.0	 1	 -30.0	 30.0;
	9	 11	 0.0	 0.208	 0.0	 65.0	 65.0	 65.0	 0.0	 0.0	 1	 -30.0	 30.0;
	9	 10	 0.0	 0.11	 0.0	 65.0	 65.0	 65.0	 0.0	 0.0	 1	 -30.0	 30.0;
	4	 12	 0.0	 0.256	 0.0	 65.0	 65.0	 65.0	 0.0	 0.0	 1	 -30.0	 30.0;
	12	 13	 0.0	 0.14	 0.0	 65.0	 65.0	 65.0	 0.0	 0.0	 1	 -30.0	 30.0;
	12	 14	 0.1231	 0.2559	 0.0	 32.0	 32.0	 32.0	 0.0	 0.0	 1	 -30.0	 30.0;
	12	 15	 0.0662	 0.1304	 0.0	 32.0	 32.0	 32.0	 0.0	 0.0	 1	 -30.0	 30.0;
	12	 16	 0.0945	 0.1987	 0.0	 32.0	 32.0	 32.0	 0.0	 0.0	 1	 -30.0	 30.0;
	14	 15	 0.221	 0.1997	 0.0	 16.0	 16.0	 16.0	 0.0	 0.0	 1	 -30.0	 30.0;
	16	 17	 0.0824	 0.1932	 0.0	 16.0	 16.0	 16.0	 0.0	 0.0	 1	 -30.0	 30.0;
	15	 18	 0.107	 0.2185	 0.0	 16.0	 16.0	 16.0	 0.0	 0.0	 1	 -30.0	 30.0;
	18	 19	 0.0639	 0.1292	 0.0	 16.0	 16.0	 16.0	 0.0	 0.0	 1	 -30.0	 30.0;
	19	 20	 0.034	 0.068	 0.0	 32.0	 32.0	 32.0	 0.0	 0.0	 1	 -30.0	 30.0;
	10	 20	 0.0936	 0.209	 0.0	 32.0	 32.0	 32.0	 0.0	 0.0	 1	 -30.0	 30.0;
	10	 17	 0.0324	 0.0845	 0.0	 32.0	 32.0	 32.0	 0.0	 0.0	 1	 -30.0	 30.0;
	10	 21	 0.0348	 0.0749	 0.0	 32.0	 32.0	 32.0	 0.0	 0.0	 1	 -30.0	 30.0;
	10	 22	 0.0727	 0.1499	 0.0	 32.0	 32.0	 32.0	 0.0	 0.0	 1	 -30.0	 30.0;
	21	 22	 0.0116	 0.0236	 0.0	 32.0	 32.0	 32.0	 0.0	 0.0	 1	 -30.0	 30.0;
	15	 23	 0.1	 0.202	 0.0	 16.0	 16.0	 16.0	 0.0	 0.0	 1	 -30.0	 30.0;
	22	 24	 0.115	 0.179	 0.0	 16.0	 16.0	 16.0	 0.0	 0.0	 1	 -30.0	 30.0;
	23	 24	 0.132	 0.27	 0.0	 16.0	 16.0	 16.0	 0.0	 0.0	 1	 -30.0	 30.0;
	24	 25	 0.1885	 0.3292	 0.0	 16.0	 16.0	 16.0	 0.0	 0.0	 1	 -30.0	 30.0;
	25	 26	 0.2544	 0.38	 0.0	 16.0	 16.0	 16.0	 0.0	 0.0	 1	 -30.0	 30.0;
	25	 27	 0.1093	 0.2087	 0.0	 16.0	 16.0	 16.0	 0.0	 0.0	 1	 -30.0	 30.0;
	28	 27	 0.0	 0.396	 0.0	 65.0	 65.0	 65.0	 0.0	 0.0	 1	 -30.0	 30.0;
	27	 29	 0.2198	 0.4153	 0.0	 16.0	 16.0	 16.0	 0.0	 0.0	 1	 -30.0	 30.0;
	27	 30	 0.3202	 0.6027	 0.0	 16.0	 16.0	 16.0	 0.0	 0.0	 1	 -30.0	 30.0;
	29	 30	 0.2399	 0.4533	 0.0	 16.0	 16.0	 16.0	 0.0	 0.0	 1	 -30.0	 30.0;
	8	 28	 0.0636	 0.2	 0.0214	 32.0	 32.0	 32.0	 0.0	 0.0	 1	 -30.0	 30.0;
	6	 28	 0.0169	 0.0599	 0.0065	 32.0	 32.0	 32.0	 0.0	 0.0	 1	 -30.0	 30.0;
];

% INFO    : === Translation Options ===
% INFO    : Load Model:                  from file ./pglib_opf_case30_fsr.m.api.sol
% INFO    : Gen Active Capacity Model:   stat
% INFO    : Gen Reactive Capacity Model: al50ag
% INFO    : Gen Active Cost Model:       stat
% INFO    : 
% INFO    : === Load Replacement Notes ===
% INFO    : Bus 2	: Pd=21.7, Qd=12.7 -> Pd=23.71, Qd=12.70
% INFO    : Bus 3	: Pd=2.4, Qd=1.2 -> Pd=2.62, Qd=1.20
% INFO    : Bus 4	: Pd=7.6, Qd=1.6 -> Pd=8.30, Qd=1.60
% INFO    : Bus 7	: Pd=22.8, Qd=10.9 -> Pd=24.91, Qd=10.90
% INFO    : Bus 8	: Pd=30.0, Qd=30.0 -> Pd=32.78, Qd=30.00
% INFO    : Bus 10	: Pd=5.8, Qd=2.0 -> Pd=6.34, Qd=2.00
% INFO    : Bus 12	: Pd=11.2, Qd=7.5 -> Pd=12.24, Qd=7.50
% INFO    : Bus 14	: Pd=6.2, Qd=1.6 -> Pd=6.77, Qd=1.60
% INFO    : Bus 15	: Pd=8.2, Qd=2.5 -> Pd=8.96, Qd=2.50
% INFO    : Bus 16	: Pd=3.5, Qd=1.8 -> Pd=3.82, Qd=1.80
% INFO    : Bus 17	: Pd=9.0, Qd=5.8 -> Pd=9.83, Qd=5.80
% INFO    : Bus 18	: Pd=3.2, Qd=0.9 -> Pd=3.50, Qd=0.90
% INFO    : Bus 19	: Pd=9.5, Qd=3.4 -> Pd=10.38, Qd=3.40
% INFO    : Bus 20	: Pd=2.2, Qd=0.7 -> Pd=2.40, Qd=0.70
% INFO    : Bus 21	: Pd=17.5, Qd=11.2 -> Pd=19.12, Qd=11.20
% INFO    : Bus 23	: Pd=3.2, Qd=1.6 -> Pd=3.50, Qd=1.60
% INFO    : Bus 24	: Pd=8.7, Qd=6.7 -> Pd=9.51, Qd=6.70
% INFO    : Bus 26	: Pd=3.5, Qd=2.3 -> Pd=3.82, Qd=2.30
% INFO    : Bus 29	: Pd=2.4, Qd=0.9 -> Pd=2.62, Qd=0.90
% INFO    : Bus 30	: Pd=10.6, Qd=1.9 -> Pd=11.58, Qd=1.90
% INFO    : 
% INFO    : === Generator Setpoint Replacement Notes ===
% INFO    : Gen at bus 1	: Pg=40.0, Qg=115.0 -> Pg=59.0, Qg=142.0
% INFO    : Gen at bus 2	: Pg=40.0, Qg=40.0 -> Pg=0.0, Qg=-133.0
% INFO    : Gen at bus 22	: Pg=25.0, Qg=32.5 -> Pg=24.0, Qg=25.0
% INFO    : Gen at bus 27	: Pg=27.5, Qg=22.5 -> Pg=53.0, Qg=31.0
% INFO    : Gen at bus 23	: Pg=15.0, Qg=20.0 -> Pg=28.0, Qg=14.0
% INFO    : Gen at bus 13	: Pg=20.0, Qg=22.5 -> Pg=51.0, Qg=41.0
% INFO    : 
% INFO    : === Generator Reactive Capacity Atleast Setpoint Value Notes ===
% INFO    : Gen at bus 1	: Qg 142.0, Qmin -20.0, Qmax 250.0 -> Qmin -170.4, Qmax 250.0
% INFO    : Gen at bus 2	: Qg -133.0, Qmin -20.0, Qmax 100.0 -> Qmin -159.6, Qmax 159.6
% INFO    : Gen at bus 22	: Qg 25.0, Qmin -15.0, Qmax 80.0 -> Qmin -30.0, Qmax 80.0
% INFO    : Gen at bus 27	: Qg 31.0, Qmin -15.0, Qmax 60.0 -> Qmin -37.2, Qmax 60.0
% INFO    : Gen at bus 23	: Qg 14.0, Qmin -10.0, Qmax 50.0 -> Qmin -16.8, Qmax 50.0
% INFO    : Gen at bus 13	: Qg 41.0, Qmin -15.0, Qmax 60.0 -> Qmin -49.2, Qmax 60.0
% INFO    : 
% INFO    : === Generator Classification Notes ===
% INFO    : NG     6   -   100.00
% INFO    : 
% INFO    : === Generator Active Capacity Stat Model Notes ===
% INFO    : Gen at bus 1 - NG	: Pg=59.0, Pmax=80.0 -> Pmax=267   samples: 2
% INFO    : Gen at bus 2 - NG	: Pg=0.0, Pmax=80.0 -> Pmax=18   samples: 1
% INFO    : Gen at bus 22 - NG	: Pg=24.0, Pmax=50.0 -> Pmax=106   samples: 5
% INFO    : Gen at bus 27 - NG	: Pg=53.0, Pmax=55.0 -> Pmax=218   samples: 1
% INFO    : Gen at bus 23 - NG	: Pg=28.0, Pmax=30.0 -> Pmax=137   samples: 1
% INFO    : Gen at bus 13 - NG	: Pg=51.0, Pmax=40.0 -> Pmax=103   samples: 1
% INFO    : 
% INFO    : === Generator Active Capacity LB Model Notes ===
% INFO    : 
% INFO    : === Generator Reactive Capacity Atleast Max 50 Percent Active Model Notes ===
% INFO    : Gen at bus 22 - NG	: Pmax 106.0, Qmin -30.0, Qmax 80.0 -> Qmin -53.0, Qmax 80.0
% INFO    : Gen at bus 27 - NG	: Pmax 218.0, Qmin -37.2, Qmax 60.0 -> Qmin -109.0, Qmax 109.0
% INFO    : Gen at bus 23 - NG	: Pmax 137.0, Qmin -16.8, Qmax 50.0 -> Qmin -69.0, Qmax 69.0
% INFO    : Gen at bus 13 - NG	: Pmax 103.0, Qmin -49.2, Qmax 60.0 -> Qmin -52.0, Qmax 60.0
% INFO    : 
% INFO    : === Generator Bounds Update Notes ===
% INFO    : 
% INFO    : === Generator Setpoint Replacement Notes ===
% INFO    : Gen at bus 1	: Pg=59.0, Qg=142.0 -> Pg=133.5, Qg=39.8
% INFO    : Gen at bus 1	: Vg=1.0 -> Vg=1.0
% INFO    : Gen at bus 2	: Pg=0.0, Qg=-133.0 -> Pg=9.0, Qg=0.0
% INFO    : Gen at bus 2	: Vg=1.025 -> Vg=1.025
% INFO    : Gen at bus 22	: Pg=24.0, Qg=25.0 -> Pg=53.0, Qg=13.5
% INFO    : Gen at bus 22	: Vg=1.025 -> Vg=1.025
% INFO    : Gen at bus 27	: Pg=53.0, Qg=31.0 -> Pg=109.0, Qg=0.0
% INFO    : Gen at bus 27	: Vg=1.025 -> Vg=1.025
% INFO    : Gen at bus 23	: Pg=28.0, Qg=14.0 -> Pg=68.5, Qg=0.0
% INFO    : Gen at bus 23	: Vg=1.025 -> Vg=1.025
% INFO    : Gen at bus 13	: Pg=51.0, Qg=41.0 -> Pg=51.5, Qg=4.0
% INFO    : Gen at bus 13	: Vg=1.025 -> Vg=1.025
% INFO    : 
% INFO    : === Writing Matpower Case File Notes ===
