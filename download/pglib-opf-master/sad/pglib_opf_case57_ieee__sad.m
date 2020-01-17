%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%                                                                  %%%%%
%%%%    IEEE PES Power Grid Library - Optimal Power Flow - v19.05     %%%%%
%%%%          (https://github.com/power-grid-lib/pglib-opf)           %%%%%
%%%%             Benchmark Group - Small Angle Difference             %%%%%
%%%%                         09 - May - 2019                          %%%%%
%%%%                                                                  %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function mpc = pglib_opf_case57_ieee__sad
mpc.version = '2';
mpc.baseMVA = 100.0;

%% bus data
%	bus_i	type	Pd	Qd	Gs	Bs	area	Vm	Va	baseKV	zone	Vmax	Vmin
mpc.bus = [
	1	 3	 55.0	 17.0	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	2	 2	 3.0	 88.0	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	3	 2	 41.0	 21.0	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	4	 1	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	5	 1	 13.0	 4.0	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	6	 2	 75.0	 2.0	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	7	 1	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	8	 2	 150.0	 22.0	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	9	 2	 121.0	 26.0	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	10	 1	 5.0	 2.0	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	11	 1	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	12	 2	 377.0	 24.0	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	13	 1	 18.0	 2.3	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	14	 1	 10.5	 5.3	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	15	 1	 22.0	 5.0	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	16	 1	 43.0	 3.0	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	17	 1	 42.0	 8.0	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	18	 1	 27.2	 9.8	 0.0	 10.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	19	 1	 3.3	 0.6	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	20	 1	 2.3	 1.0	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	21	 1	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	22	 1	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	23	 1	 6.3	 2.1	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	24	 1	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	25	 1	 6.3	 3.2	 0.0	 5.9	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	26	 1	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	27	 1	 9.3	 0.5	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	28	 1	 4.6	 2.3	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	29	 1	 17.0	 2.6	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	30	 1	 3.6	 1.8	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	31	 1	 5.8	 2.9	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	32	 1	 1.6	 0.8	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	33	 1	 3.8	 1.9	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	34	 1	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	35	 1	 6.0	 3.0	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	36	 1	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	37	 1	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	38	 1	 14.0	 7.0	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	39	 1	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	40	 1	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	41	 1	 6.3	 3.0	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	42	 1	 7.1	 4.4	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	43	 1	 2.0	 1.0	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	44	 1	 12.0	 1.8	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	45	 1	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	46	 1	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	47	 1	 29.7	 11.6	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	48	 1	 0.0	 0.0	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	49	 1	 18.0	 8.5	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	50	 1	 21.0	 10.5	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	51	 1	 18.0	 5.3	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	52	 1	 4.9	 2.2	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	53	 1	 20.0	 10.0	 0.0	 6.3	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	54	 1	 4.1	 1.4	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	55	 1	 6.8	 3.4	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	56	 1	 7.6	 2.2	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
	57	 1	 6.7	 2.0	 0.0	 0.0	 1	    1.00000	    0.00000	 1.0	 1	    1.06000	    0.94000;
];

%% generator data
%	bus	Pg	Qg	Qmax	Qmin	Vg	mBase	status	Pmax	Pmin
mpc.gen = [
	1	 122.5	 0.0	 123.0	 -123.0	 1.0	 100.0	 1	 245.0	 0.0;
	2	 0.0	 16.5	 50.0	 -17.0	 1.0	 100.0	 1	 0.0	 0.0;
	3	 30.0	 10.0	 30.0	 -10.0	 1.0	 100.0	 1	 60.0	 0.0;
	6	 0.0	 8.5	 25.0	 -8.0	 1.0	 100.0	 1	 0.0	 0.0;
	8	 579.5	 30.0	 200.0	 -140.0	 1.0	 100.0	 1	 1159.0	 0.0;
	9	 0.0	 3.0	 9.0	 -3.0	 1.0	 100.0	 1	 0.0	 0.0;
	12	 259.5	 2.5	 155.0	 -150.0	 1.0	 100.0	 1	 519.0	 0.0;
];

%% generator cost data
%	2	startup	shutdown	n	c(n-1)	...	c0
mpc.gencost = [
	2	 0.0	 0.0	 3	   0.000000	  16.960624	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	   0.000000	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	  34.075557	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	   0.000000	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	  30.441037	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	   0.000000	   0.000000;
	2	 0.0	 0.0	 3	   0.000000	  37.188979	   0.000000;
];

%% branch data
%	fbus	tbus	r	x	b	rateA	rateB	rateC	ratio	angle	status	angmin	angmax
mpc.branch = [
	1	 2	 0.0083	 0.028	 0.129	 1005.0	 1005.0	 1005.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	2	 3	 0.0298	 0.085	 0.0818	 326.0	 326.0	 326.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	3	 4	 0.0112	 0.0366	 0.038	 767.0	 767.0	 767.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	4	 5	 0.0625	 0.132	 0.0258	 201.0	 201.0	 201.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	4	 6	 0.043	 0.148	 0.0348	 191.0	 191.0	 191.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	6	 7	 0.02	 0.102	 0.0276	 283.0	 283.0	 283.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	6	 8	 0.0339	 0.173	 0.047	 167.0	 167.0	 167.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	8	 9	 0.0099	 0.0505	 0.0548	 570.0	 570.0	 570.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	9	 10	 0.0369	 0.1679	 0.044	 171.0	 171.0	 171.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	9	 11	 0.0258	 0.0848	 0.0218	 331.0	 331.0	 331.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	9	 12	 0.0648	 0.295	 0.0772	 98.0	 98.0	 98.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	9	 13	 0.0481	 0.158	 0.0406	 178.0	 178.0	 178.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	13	 14	 0.0132	 0.0434	 0.011	 647.0	 647.0	 647.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	13	 15	 0.0269	 0.0869	 0.023	 323.0	 323.0	 323.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	1	 15	 0.0178	 0.091	 0.0988	 317.0	 317.0	 317.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	1	 16	 0.0454	 0.206	 0.0546	 140.0	 140.0	 140.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	1	 17	 0.0238	 0.108	 0.0286	 266.0	 266.0	 266.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	3	 15	 0.0162	 0.053	 0.0544	 530.0	 530.0	 530.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	4	 18	 0.0	 0.555	 0.0	 53.0	 53.0	 53.0	 0.97	 0.0	 1	 -4.943905	 4.943905;
	4	 18	 0.0	 0.43	 0.0	 69.0	 69.0	 69.0	 0.978	 0.0	 1	 -4.943905	 4.943905;
	5	 6	 0.0302	 0.0641	 0.0124	 414.0	 414.0	 414.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	7	 8	 0.0139	 0.0712	 0.0194	 405.0	 405.0	 405.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	10	 12	 0.0277	 0.1262	 0.0328	 228.0	 228.0	 228.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	11	 13	 0.0223	 0.0732	 0.0188	 384.0	 384.0	 384.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	12	 13	 0.0178	 0.058	 0.0604	 484.0	 484.0	 484.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	12	 16	 0.018	 0.0813	 0.0216	 353.0	 353.0	 353.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	12	 17	 0.0397	 0.179	 0.0476	 160.0	 160.0	 160.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	14	 15	 0.0171	 0.0547	 0.0148	 512.0	 512.0	 512.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	18	 19	 0.461	 0.685	 0.0	 36.0	 36.0	 36.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	19	 20	 0.283	 0.434	 0.0	 57.0	 57.0	 57.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	21	 20	 0.0	 0.7767	 0.0	 38.0	 38.0	 38.0	 1.043	 0.0	 1	 -4.943905	 4.943905;
	21	 22	 0.0736	 0.117	 0.0	 213.0	 213.0	 213.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	22	 23	 0.0099	 0.0152	 0.0	 1617.0	 1617.0	 1617.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	23	 24	 0.166	 0.256	 0.0084	 97.0	 97.0	 97.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	24	 25	 0.0	 1.182	 0.0	 25.0	 25.0	 25.0	 1.0	 0.0	 1	 -4.943905	 4.943905;
	24	 25	 0.0	 1.23	 0.0	 24.0	 24.0	 24.0	 1.0	 0.0	 1	 -4.943905	 4.943905;
	24	 26	 0.0	 0.0473	 0.0	 621.0	 621.0	 621.0	 1.043	 0.0	 1	 -4.943905	 4.943905;
	26	 27	 0.165	 0.254	 0.0	 97.0	 97.0	 97.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	27	 28	 0.0618	 0.0954	 0.0	 259.0	 259.0	 259.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	28	 29	 0.0418	 0.0587	 0.0	 408.0	 408.0	 408.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	7	 29	 0.0	 0.0648	 0.0	 453.0	 453.0	 453.0	 0.967	 0.0	 1	 -4.943905	 4.943905;
	25	 30	 0.135	 0.202	 0.0	 121.0	 121.0	 121.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	30	 31	 0.326	 0.497	 0.0	 50.0	 50.0	 50.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	31	 32	 0.507	 0.755	 0.0	 33.0	 33.0	 33.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	32	 33	 0.0392	 0.036	 0.0	 552.0	 552.0	 552.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	34	 32	 0.0	 0.953	 0.0	 31.0	 31.0	 31.0	 0.975	 0.0	 1	 -4.943905	 4.943905;
	34	 35	 0.052	 0.078	 0.0032	 313.0	 313.0	 313.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	35	 36	 0.043	 0.0537	 0.0016	 427.0	 427.0	 427.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	36	 37	 0.029	 0.0366	 0.0	 629.0	 629.0	 629.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	37	 38	 0.0651	 0.1009	 0.002	 245.0	 245.0	 245.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	37	 39	 0.0239	 0.0379	 0.0	 655.0	 655.0	 655.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	36	 40	 0.03	 0.0466	 0.0	 530.0	 530.0	 530.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	22	 38	 0.0192	 0.0295	 0.0	 834.0	 834.0	 834.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	11	 41	 0.0	 0.749	 0.0	 40.0	 40.0	 40.0	 0.955	 0.0	 1	 -4.943905	 4.943905;
	41	 42	 0.207	 0.352	 0.0	 72.0	 72.0	 72.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	41	 43	 0.0	 0.412	 0.0	 72.0	 72.0	 72.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	38	 44	 0.0289	 0.0585	 0.002	 450.0	 450.0	 450.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	15	 45	 0.0	 0.1042	 0.0	 282.0	 282.0	 282.0	 0.955	 0.0	 1	 -4.943905	 4.943905;
	14	 46	 0.0	 0.0735	 0.0	 400.0	 400.0	 400.0	 0.9	 0.0	 1	 -4.943905	 4.943905;
	46	 47	 0.023	 0.068	 0.0032	 409.0	 409.0	 409.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	47	 48	 0.0182	 0.0233	 0.0	 993.0	 993.0	 993.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	48	 49	 0.0834	 0.129	 0.0048	 191.0	 191.0	 191.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	49	 50	 0.0801	 0.128	 0.0	 195.0	 195.0	 195.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	50	 51	 0.1386	 0.22	 0.0	 113.0	 113.0	 113.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	10	 51	 0.0	 0.0712	 0.0	 412.0	 412.0	 412.0	 0.93	 0.0	 1	 -4.943905	 4.943905;
	13	 49	 0.0	 0.191	 0.0	 154.0	 154.0	 154.0	 0.895	 0.0	 1	 -4.943905	 4.943905;
	29	 52	 0.1442	 0.187	 0.0	 125.0	 125.0	 125.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	52	 53	 0.0762	 0.0984	 0.0	 236.0	 236.0	 236.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	53	 54	 0.1878	 0.232	 0.0	 99.0	 99.0	 99.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	54	 55	 0.1732	 0.2265	 0.0	 103.0	 103.0	 103.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	11	 43	 0.0	 0.153	 0.0	 192.0	 192.0	 192.0	 0.958	 0.0	 1	 -4.943905	 4.943905;
	44	 45	 0.0624	 0.1242	 0.004	 212.0	 212.0	 212.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	40	 56	 0.0	 1.195	 0.0	 25.0	 25.0	 25.0	 0.958	 0.0	 1	 -4.943905	 4.943905;
	56	 41	 0.553	 0.549	 0.0	 38.0	 38.0	 38.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	56	 42	 0.2125	 0.354	 0.0	 72.0	 72.0	 72.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	39	 57	 0.0	 1.355	 0.0	 22.0	 22.0	 22.0	 0.98	 0.0	 1	 -4.943905	 4.943905;
	57	 56	 0.174	 0.26	 0.0	 94.0	 94.0	 94.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	38	 49	 0.115	 0.177	 0.003	 139.0	 139.0	 139.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	38	 48	 0.0312	 0.0482	 0.0	 511.0	 511.0	 511.0	 0.0	 0.0	 1	 -4.943905	 4.943905;
	9	 55	 0.0	 0.1205	 0.0	 244.0	 244.0	 244.0	 0.94	 0.0	 1	 -4.943905	 4.943905;
];

% INFO    : === Translation Options ===
% INFO    : Phase Angle Bound:           4.943905 (deg.)
% INFO    : 
% INFO    : === Generator Bounds Update Notes ===
% INFO    : 
% INFO    : === Base KV Replacement Notes ===
% INFO    : 
% INFO    : === Transformer Setting Replacement Notes ===
% INFO    : 
% INFO    : === Line Capacity Monotonicity Notes ===
% INFO    : 
% INFO    : === Writing Matpower Case File Notes ===
