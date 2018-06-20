libname data "N:\IRSDept\IRSUser\DI\Technical Modelling\Projects\2016\20160907_Pricing_Modernisation\data\ClaimsCost\Ex-DI\Motor";

data policy;
	set data.policy;
	where dt_pinfr > '01Jan2015'd;
run;

proc sort data= c_cxcar.changex(keep=nm_polh dt_pinfr ct_drvexp)
	out=drvxp(index=(xp_key = (nm_polh dt_pinfr))) nodupkey;
	by nm_polh dt_pinfr ct_drvexp;
run;

proc datasets lib=work;
	modify drvxp;
	index create xp_key = (nm_polh dt_pinfr);
quit;

proc sort data = policy;
	by nm_polh descending dt_pinfr;
run;

data policy2;
	set policy(obs=max);
	by nm_polh descending dt_pinfr;
	set drvxp(keep=nm_polh dt_pinfr ct_drvexp) key=xp_key /unique;
	if _error_ then do;
		_error_=0;
		if first.nm_polh then call missing(ct_drvexp);
	end;
run;

data test;
	set policy2;
	where missing(ct_drvexp);
run;
/**/
/*data claims(index=(nm_polh));*/
/*	set data.claims;*/
/*	where incdate > '01Jan2015'd and not missing(nm_polh);*/
/*run;*/
/**/
/*proc sort data=claims out=nodup_claimno nodupkey;*/
/*	by nm_polh;*/
/*run;*/
/**/
/*data noclaim_pol*/
/*	 claim_pol;*/
/*	 merge policy2(in=a)*/
/*	 		nodup_claimno(in=b)*/
/*			;*/
/*	by nm_polh;*/
/*	if a and not b then output noclaim_pol;*/
/*	else if a and b then output claim_pol;*/
/*run;*/
/**/
/*proc sql;*/
/*	create table claim_pol2*/
/*	as select*/
/*	* from*/
/*	claim_pol as a */
/*	left join*/
/*	claims as b*/
/*	on a.nm_polh = b.nm_polh and  a.dt_pinfr <= b.incdate <= a.dt_pinto_o*/
/*	;*/
/*run;*/
/*quit;*/
