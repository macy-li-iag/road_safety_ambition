#libs
library(data.table)
library(fst)
library(magrittr)
library(ggplot2)
library(scales)
library(plotly)
library(rmarkdown)
#setup directories
workdir<-"C:/road_safety_ambition/driver experience"
dataloc<-"T:/Analytics/Ad_hoc/20180620_road_safety_ambition_intiatives/driver experience/"
setwd(workdir)

#read in data
policydat<-fread(paste0(dataloc,"policydat.csv"))
write.fst(policydat,paste0(dataloc,"policydat.fst"))
clmdat<-fread("N:/IRSDept/IRSUser/DI/Technical Modelling/Projects/2016/20160907_Pricing_Modernisation/data/ClaimsCost/Ex-DI/Motor/claims.csv")
write.fst(clmdat,paste0(dataloc,"clmdat.fst"))

#clean up datatsets for polclm
policydat[,":="(dt_pinfr=as.Date(DT_PINFR,"%d%b%Y"),
                dt_pinto_o=as.Date(DT_PINTO_O,"%d%b%Y"))]
policydat<-policydat[dt_pinto_o >=dt_pinfr & NM_POLH!=""]
clmno<-clmdat$nm_polh %>% unique
clmpol<-policydat[NM_POLH %in% clmno]

clmdat[,":="(incdate_=as.Date(incdate,"%d/%m/%Y")
             ,incdate=as.Date(incdate,"%d/%m/%Y"))]

setkey(clmpol,NM_POLH,dt_pinfr,dt_pinto_o)


# summarise claims to be merged

polclm<-foverlaps(clmdat,clmpol
                  ,by.x=c("nm_polh","incdate","incdate_")
                  ,by.y=c("NM_POLH","dt_pinfr","dt_pinto_o")
                  ,type="within"
                  ,nomatch=NA
                  ,which=FALSE
                  )


polclm<-polclm[!is.na(DT_PINFR)]

polclm2<-polclm[nat_peril_flag==0,.(clm=sum(clm_cs)
                                    ,cs=sum(clmcost_net_gst_net_excess)
                                    ,si=sum(as.numeric(gsub("\\$|,","",SM_TOTAL)))
                                    ),by=.(nm_polh,dt_pinfr)]
  
#check for dups
polclm %>% nrow
polclm$claimno %>% unique %>% length
polclm[,.(nm_polh,dt_pinfr)] %>% unique %>% nrow

polclm3 <- merge(policydat,polclm2,by.x=c('NM_POLH','dt_pinfr'),by.y=c("nm_polh","dt_pinfr"),all.x=T)
polclm3 %>% nrow
policydat %>% nrow

#construct graphs
polclm3<-polclm3 %>% 
  .[,exposure:=(difftime(dt_pinto_o,dt_pinfr,units='days')+1)/365]

#driver age
plotdata<-function(var){
  plotdat<-polclm3[,.(N=sum(as.numeric(exposure))
                       ,claimcount=sum(clm,na.rm=T)
                       ,claimcost=sum(cs,na.rm=T)
                       ,totalsi=sum(si,na.rm=T))
                    ,by=.(var=get(var))] %>%
    .[order(var)] %>%
    .[,":="(claimfreq=claimcount/N
            ,acs=claimcost/claimcount
            ,cpp=claimcost/N
            ,prop_claim=claimcount/sum(claimcount)
            ,prop_exp=N/sum(N))]
  
  scale1=max(plotdat[claimcount>5000]$claimfreq)/max(plotdat[claimcount>5000]$prop_exp)
  scale2=max(plotdat[claimcount>5000]$acs)/max(plotdat[claimcount>5000]$prop_claim)
  return(list(plotdat,scale1,scale2))
  }


#plot
age<-plotdata("CTAGEYD")

plot1<-ggplot(age[[1]])+
  geom_col(aes(x=var,y=prop_exp*age[[2]]),alpha=0.2)+
  geom_line(aes(x=var,y=claimfreq))+
  scale_x_continuous(limits=c(16,80))+
  scale_y_continuous(labels=percent,limits=c(0,0.25)
                     ,sec.axis=sec_axis(~./age[[2]],breaks=seq(0,1,by=0.005),labels=percent,name="proportion of exposure"))+
  labs(x="age of youngest driver",y="claim frequency")+
  theme_bw()

plot2<-ggplot(age[[1]])+
  geom_col(aes(x=var,y=prop_claim*age[[3]]),alpha=0.2)+
  geom_line(aes(x=var,y=acs))+
  scale_x_continuous(limits=c(16,80))+
  scale_y_continuous(labels=dollar
                     ,limits=c(0,5000)
                     ,sec.axis=sec_axis(~./age[[3]],breaks=seq(0,1,by=0.005),labels=percent,name="proportion of total claims"))+
  labs(x="age of youngest driver",y="average cost per claim")+
  theme_bw()

#driver exp
drvexp<-plotdata("drvexp")
#plot

plotexp<-ggplot(drvexp[[1]])+
  geom_col(aes(x=var,y=prop_exp*drvexp[[2]]),alpha=0.2)+
  geom_line(aes(x=var,y=claimfreq))+
  scale_x_continuous(limits=c(0,50))+
  scale_y_continuous(labels=percent,limits=c(0,0.25)
                     ,sec.axis=sec_axis(~./drvexp[[2]],breaks=seq(0,1,by=0.005),labels=percent,name="proportion of exposure"))+
  labs(x="years of experience",y="claim frequency")+
  theme_bw()

plotexp2<-ggplot(drvexp[[1]])+
  geom_col(aes(x=var,y=prop_claim*drvexp[[3]]),alpha=0.2)+
  geom_line(aes(x=var,y=acs))+
  scale_x_continuous(limits=c(0,50))+
  scale_y_continuous(labels=dollar
                     # ,limits=c(0,6000)
                     ,sec.axis=sec_axis(~./drvexp[[3]],breaks=seq(0,1,by=0.005),labels=percent,name="proportion of total claims"))+
  labs(x="years of experience",y="average cost per claim")+
  theme_bw()

#zoom in on low experience group and bucket by experience bands
drvexp_age<-polclm3[,.(N=sum(as.numeric(exposure))
                       ,claimcount=sum(clm,na.rm=T)
                       ,claimcost=sum(cs,na.rm=T)
                       ,totalsi=sum(si,na.rm=T))
                    # ,by=.(exp_bucket=cut(drvexp,breaks=c(seq(0,50,by=5),100),include.lowest = T),CTAGEYD)
                    ,by=.(exp_bucket=drvexp,CTAGEYD)
                    ] %>%
  .[order(exp_bucket,CTAGEYD)] %>%
  .[,":="(claimfreq=claimcount/N
          ,acs=claimcost/claimcount
          ,cpp=claimcost/N
          ,prop_claim=claimcount/sum(claimcount)
          ,prop_exp=N/sum(N)
          ,cds=claimcost/totalsi
          )]


scale1_expage=max(drvexp_age[exp_bucket<9 &claimcount>100]$claimfreq)/max(drvexp_age[exp_bucket<9 &claimcount>100]$prop_exp)
scale2_expage=max(drvexp_age[exp_bucket<9 &claimcount>100]$acs)/max(drvexp_age[exp_bucket<9 & claimcount>100]$prop_claim)

#plot
plot_drvexp_age<-ggplot(drvexp_age[exp_bucket<10 &claimcount>100])+
  geom_col(aes(x=CTAGEYD,y=prop_exp*scale1_expage),alpha=0.2)+
  geom_line(aes(x=CTAGEYD,y=claimfreq))+
  scale_x_continuous(breaks=seq(0,100,by=5))+
  scale_y_continuous(labels=percent,limits=c(0,0.3)
  #                    ,sec.axis=sec_axis(~./scale1_expage*scale2_expage,breaks=seq(0,1,by=0.005),labels=percent,name="proportion of exposure")
                     )+
  labs(x="driver age",y="claim frequency")+
  facet_wrap(~exp_bucket,ncol=3)+
  theme_bw()
plot_drvexp_age

plot_drvexp_age_cs<-ggplot(drvexp_age[exp_bucket<9 & claimcount>100])+
  geom_col(aes(x=CTAGEYD,y=prop_claim*scale2_expage),alpha=0.2)+
  geom_line(aes(x=CTAGEYD,y=acs))+
  scale_x_continuous(breaks=seq(0,100,by=5))+
  # scale_y_continuous(labels=dollar,limits=c(0,0.5)
  #                    ,sec.axis=sec_axis(~./scale1_expage*scale2_expage,breaks=seq(0,1,by=0.005),labels=percent,name="proportion of exposure")
                     # )+
  labs(x="driver age",y="average cost per claim")+
  facet_wrap(~exp_bucket,ncol=3)+
  theme_bw()
plot_drvexp_age_cs

render(input="findings.rmd"
       ,output_file=paste0(dataloc,"output/deck.html"))

