SWOF
-- Column 1: water saturation
--   	     - this has been set to (almost) equally spaced values from 0.12 to 1
-- Column 2: water relative permeability
--   	     - generated from the Corey-type approx. formula
--	       the coeffisient is set to 10e-5, S_{orw}=0 and S_{wi}=0.12
-- Column 3: oil relative permeability when only oil and water are present
--	     - we will use the same values as in column 3 in SGOF.
-- 	       This is not really correct, but since only the first 
--	       two values are of importance, this does not really matter
-- Column 4: water-oil capillary pressure (psi) 

0.12	0    		 	1	0
0.18	4.64876033057851E-008	1	0
0.24	0.000000186		0.997	0
0.3	4.18388429752066E-007	0.98	0
0.36	7.43801652892562E-007	0.7	0
0.42	1.16219008264463E-006	0.35	0
0.48	1.67355371900826E-006	0.2	0
0.54	2.27789256198347E-006	0.09	0
0.6	2.97520661157025E-006	0.021	0
0.66	3.7654958677686E-006	0.01	0
0.72	4.64876033057851E-006	0.001	0
0.78	0.000005625		0.0001	0
0.84	6.69421487603306E-006	0	0
0.91	8.05914256198347E-006	0	0
1	0.00001			0	0 /


SGOF
-- Column 1: gas saturation
-- Column 2: gas relative permeability
-- Column 3: oil relative permeability when oil, gas and connate water are present
-- Column 4: oil-gas capillary pressure (psi)
-- 	     - stated to be zero in Odeh's paper

-- Values in column 1-3 are taken from table 3 in Odeh's paper:
0	0	1	0
0.001	0	1	0
0.02	0	0.997	0
0.05	0.005	0.980	0
0.12	0.025	0.700	0
0.2	0.075	0.350	0
0.25	0.125	0.200	0
0.3	0.190	0.090	0
0.4	0.410	0.021	0
0.45	0.60	0.010	0
0.5	0.72	0.001	0
0.6	0.87	0.0001	0
0.7	0.94	0.000	0
0.85	0.98	0.000	0 
0.88	0.984	0.000	0 /
--1.00	1.0	0.000	0 /
-- Warning from Eclipse: first sat. value in SWOF + last sat. value in SGOF
-- 	   		 must not be greater than 1, but Eclipse still runs
-- Flow needs the sum to be excactly 1 so I added a row with gas sat. =  0.88
-- The corresponding krg value was estimated by assuming linear rel. between
-- gas sat. and krw. between gas sat. 0.85 and 1.00 (the last two values given)
