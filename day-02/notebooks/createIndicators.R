createIndicators <- function(price, p = 16){
  
  require(TTR)
  require(dplyr)
  require(magrittr)
  
  adx <- ADX(price[,c("High","Low","Close")], n = 16) %>% as.data.frame %>%  mutate(.,oscDX = DIp -DIn) %>% transmute(.,DX, ADX, oscDX) %>% as.matrix()
  
  ar <- aroon(price[,c('High', 'Low')], n = p) %>% extract(,3)
  
  atr <- ATR(price[,c("High","Low","Close")], n = p, maType = "EMA") %>% extract(,1:2)
  
  cci <- CCI(price[,c("High","Low","Close")], n = p)
  
  chv <- chaikinVolatility(price[,c("High","Low","Close")], n = p)
  
  cmo <- CMO(price[ ,'Med'], n = p)
  
  macd <- MACD(price[ ,'Med'], 12, 26, 9) %>% as.data.frame() %>% mutate(., vsig = signal %>% diff %>% c(NA,.) %>% multiply_by(10)) %>% transmute(., sign = signal, vsig) %>% as.matrix()
  
  rsi <- RSI(price[ ,'Med'], n = p)
  
  stoh <- stoch(price[,c("High","Low","Close")], nFastK = p, nFastD =3, nSlowD = 3, maType = "EMA")%>% as.data.frame() %>% 
          mutate(., oscK = fastK - fastD)%>% transmute(.,slowD, oscK)%>% as.matrix()
  
  smi <- SMI(as.matrix(price[,c("High","Low","Close")]),n = p, nFast = 2, nSlow = 25, nSig = 9)
  
  vol <- volatility(as.matrix(price[,c("Med", "High","Low","Close")]), n = p, calc = "yang.zhang", N = 144)
  Indicators<- cbind(adx, ar, atr, cci, chv, cmo, macd, rsi, stoh, smi, vol)
  return(Indicators)
}
