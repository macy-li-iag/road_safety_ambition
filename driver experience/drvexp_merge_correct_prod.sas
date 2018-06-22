*Source Anthony Clissold;
libname huon "k:\IRSData\NIGL\Data\CRDB\Motor\HUON\2018-04\";

proc sql; select count(distinct nm_polh) from huon.dpt031 where nm_polh in (select nm_polh from c_car.change where dt_pinfr > '01Jan2015'd); quit;

/*Merge driving exp table and identity to get driver dob*/
proc sql;
  create table dpt031 as  
  select a.*, b.dtbirth 
  from huon.dpt031 as a
    left join huon.identity as b
    on a.nm_ident=b.nm_ident
  where a.nm_polh in (select nm_polh from c_car.change where dt_pinfr > '01Jan2015'd);
quit;

proc sql; select count(distinct nm_polh) from dpt031; quit;

/*Create and merge to change file by pol num and trans num to get dt_pinfr*/
data chg;
  set c_car.change(where=( dt_pinfr > '01Jan2015'd/*and dt_pinfr > '01MAY2016'd*/));
/*  c_pdv.change(where=(cdsubco = 300 and dt_pinfr > '01MAY2016'd and cdtyprdh = 'BKCP'));*/
run;

proc sql;
  create table drvexp_dat as  
  select chg.*, a.* 
  from chg
    left join dpt031 as a on chg.nm_polh=a.nm_polh and chg.nm_btran=a.nm_btran
/*    where chg.nm_polh in (select nm_polh from c_car.change where cdsubco = 300)*/
  order by chg.nm_polh, a.nm_ident, chg.dt_pinfr;
quit;

/*proc sql; select count(distinct nm_polh) from drvexp_dat; quit;*/

/*Keep only rows with the most recent transaction*/
proc sql;
  create table drvexp_dat_trans as  
  select nm_polh, dt_pinfr, max(nm_btran) as max_tran
  from drvexp_dat
  group by nm_polh, dt_pinfr;
quit;

proc sql;
  create table drvexp_dat2 as  
  select a.*, b.max_tran 
  from drvexp_dat as a
    left join drvexp_dat_trans as b
    on a.nm_polh=b.nm_polh and a.dt_pinfr=b.dt_pinfr
  where a.nm_btran=b.max_tran;
quit;

/*proc sql; select count(distinct nm_polh) from drvexp_dat2; quit;*/

/*Keep only rows related to the youngest driver on the policy*/
proc sql;
  create table drvexp_dat_yd as  
  select nm_polh, dt_pinfr, max(dtbirth) as yddob format=ddmmyy10. 
  from drvexp_dat2
  where dtbirth ne .
  group by nm_polh, dt_pinfr;
quit;

proc sql;
  create table drvexp_dat3 as  
  select a.*, b.yddob 
  from drvexp_dat2 as a
    left join drvexp_dat_yd as b
    on a.nm_polh=b.nm_polh and a.dt_pinfr=b.dt_pinfr
  where a.dtbirth=b.yddob;
quit;

/*proc sql; select count(distinct nm_polh) from drvexp_dat3; quit;*/

/*Keep min drvexp where multiple youngest drivers*/
proc sql;
  create table drvexp_dat4 as  
  select nm_polh, dt_pinfr, nm_btran, dtbirth, yddob, min(ctdrvexp) as drvexp 
  from drvexp_dat3
  group by nm_polh, dt_pinfr, nm_btran, dtbirth, yddob;
quit;

/*Merge driving experience onto predrenx file*/
proc sql;
  create table predrenx1a as  
  select a.*, b.drvexp
  from chg as a 
    left join drvexp_dat4 as b on a.nm_polh=b.nm_polh and a.dt_pinfr=b.dt_pinfr;
quit;

/*proc sql; select count(distinct nm_polh) from predrenx1a(where=(drvexp ne .)); quit;*/
/*proc sql; select count(distinct nm_polh) from predrenx1a(where=(drvexp=.)); quit;*/

/*******For missing values from first merge*******/

/*Keep only rows with the latest effective date*/
proc sql;
  create table dpt031_missing_chto as  
  select nm_polh, max(dt_chto) as max_chto format=ddmmyy10.
  from dpt031
  where nm_polh in (select nm_polh from predrenx1a where drvexp = .)
  group by nm_polh;
quit;

proc sql;
  create table dpt031a as  
  select a.*, b.max_chto 
  from dpt031 as a
    left join dpt031_missing_chto as b
    on a.nm_polh=b.nm_polh
  where a.dt_chto=b.max_chto;
quit;

/*proc sql; select count(distinct nm_polh) from dpt031_missing; quit;*/

/*Keep only rows related to the youngest driver*/
proc sql;
  create table drvexp_dat_missing_yd as  
  select nm_polh, dt_effch, max(dtbirth) as yddob format=ddmmyy10. 
  from dpt031a
  where dtbirth ne .
  group by nm_polh, dt_effch;
quit;

proc sql;
  create table drvexp_dat_missing2 as  
  select a.*, b.yddob 
  from dpt031a as a
    left join drvexp_dat_missing_yd as b
    on a.nm_polh=b.nm_polh and a.dt_effch=b.dt_effch
  where a.dtbirth=b.yddob;
quit;

/*proc sql; select count(distinct nm_polh) from drvexp_dat_missing2; quit;*/

/*Keep min drvexp where multiple youngest drivers*/
proc sql;
  create table drvexp_dat_missing3 as  
  select nm_polh, dt_chto, nm_btran, dtbirth, yddob, min(ctdrvexp) as drvexp 
  from drvexp_dat_missing2
  group by nm_polh, dt_chto, nm_btran, dtbirth, yddob;
quit;

/*Merge driving experience onto predrenx file*/
proc sql;
  create table predrenx1a_missing as  
  select a.*, b.drvexp
  from chg as a 
    left join drvexp_dat_missing3 as b on a.nm_polh=b.nm_polh
  where a.nm_polh in (select nm_polh from predrenx1a where drvexp = .);
quit;

/*proc sql; select count(distinct nm_polh) from predrenx1a_missing(where=(drvexp ne .)); quit;*/
/*proc sql; select count(distinct nm_polh) from predrenx1a_missing(where=(drvexp = .)); quit;*/

/*Stack two subsets back together and replace variables*/
/*If neither fix worked (6 pols) then take drvexp to be years since 18*/
data predrenx1b;
  set predrenx1a(where=(drvexp ne .) in=a) predrenx1a_missing(in=b);

  if a then err = 0;
    else if b and drvexp ne . then err = 1;
    else err = 2;
    
/*  ct_drvexp = drvexp;*/
/*  if drvexp = . then ct_drvexp = sum(ctageyd,-18);*/
/*  if drvexp = . then drvexp = sum(ctageyd,-18);*/
run;
proc freq data=predrenx1b; tables err; run;

/*case when catchment_area_suburb eq "" then catchment_area_postcode else catchment_area_suburb end as catchment_area_final,*/

proc sql;
  create table &out. as  
  select a.*,
  case when a.cdsubco eq 300 then b.ct_drvexp else a.ct_drvexp end as ct_drvexp
  from c_car.change as a 
    left join predrenx1b as b on a.nm_polh=b.nm_polh and a.dt_pinfr=b.dt_pinfr;
quit;

/*proc sql; select count(distinct nm_polh) from predrenx1b(where=(drvexp ne .)); quit;*/
/*proc sql; select count(distinct nm_polh) from predrenx1b(where=(drvexp = .)); quit;*/

/*proc sort data=predrenx1b; by nm_polh dt_pinfr; run;*/
/*proc sort data=predrenx1b; by nm_polh dt_pinfr; run;*/

/************************************************************************************************/
proc sql;
  create table compare_pred as  
  select a.*, b.ct_drvexp as ct_drvexp_new 
  from predrenx1d_old as a
    left join predrenx1b as b on a.nm_polh=b.nm_polh and a.dt_pinfr=b.dt_pinfr;
quit;

proc freq data=compare_pred; tables ct_drvexp_new ct_drvexp; run;

data compare_pred2;
  set compare_pred;
  where (ct_drvexp ne ct_drvexp_new) and (ct_drvexp+1 ne ct_drvexp_new or (ct_drvexp+1=ct_drvexp_new and dt_pinfr >= '01oct2016'd));
/*  keep nm_polh dt_pinfr ct_drvexp ct_drvexp_new;*/
run;

data compare_pred3;
  set compare_pred2;
  where (ctageyd >= 19 and ct_drvexp = 0) or (ctageyd >= 20 and ct_drvexp < 2) or (ctageyd >= 21 and ct_drvexp < 3) or (ctageyd >= 22 and ct_drvexp < 4);
run;



proc freq data=compare_pred(where=(ct_drvexp+1 = ct_drvexp_new)); tables dt_pinfr; run;
