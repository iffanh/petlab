-- Generated [
-- Format      : ECLIPSE keywords (ASCII)
-- Exported by : Petrel 2017.4 Schlumberger
-- User name   : Aqnan Matthew
-- Date        : Tuesday, April 13 2021 22:33:26
-- Project     : EggModel.pet
-- Generated ]

--RPTSOL
--    RESTART=2 FIP=3/

--EQUIL
-- 
--Equilibration Data Specification
--      1850     320    2028       0       0       0       1       0       0
--/

RPTRST
  BASIC=2 PRES SGAS SOIL SWAT DENG  DENW  DENO  ZMF BGAS BOIL BWAT/

--RPTSOL
--  RESTART=2 /
/

--RESTART                                -- Generated : Petrel
--  '/WAG_BASE' 96 /

INCLUDE
'/home/iffanh/tests/with_Jinjie/petlab/data/Egg_CCS/WAG_BASE_RESTART/WAG_BASE_PRESSUREINIT.GRDECL' /

INCLUDE
'/home/iffanh/tests/with_Jinjie/petlab/data/Egg_CCS/WAG_BASE_RESTART/WAG_BASE_SWATINIT.GRDECL' /

INCLUDE
'/home/iffanh/tests/with_Jinjie/petlab/data/Egg_CCS/WAG_BASE_RESTART/WAG_BASE_SGASINIT.GRDECL' /

ZMF                                  -- Generated : Petrel
  25200*0.029910303
  25200*0.294293029 
  25200*0.137401014
  25200*0.153182015 
  25200*0.20016202 
  25200*0.129591013
  25200*0.055460606/
