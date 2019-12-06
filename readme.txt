pt.sh skriptin avulla voidaan kätevästi kohdistaa kansiollinen äänitteitä CSC:n
palvelimilla. Skriptiä tulee muokata tähdillä merkattujen kohtien osalta
vastaamaan omia käyttäjän tietoja.


testbench.py sisältää suurimman osan työssä käytetystä koodista. Moduulia voi
käyttää komentorivin kautta. Ohjeet cl-käyttöön tulostetaan automaattisesti.

Modulin funktiot on tarkoitettu datan manipulointiin, testaamiseen ja mallien
opettamiseen:

    MALLINOPETUS
train sovittaa linreg mallin yhdistetyillä kriteereillä (opetusjoukko on aina
sama!)
trainMultiple sovittaa kaikki kriteerit omaan lineaariseen regressiomalliinsa
ja palauttaa mallit


    TESTAUS
testRandom testaa linreg mallia yhdistetyillä arvoilla ja 10 eri jaolla
testFeaturesMultiple testaa eri piirrekombinaatioilla jokaista kriteeriä
classifierTest testaa kaikkia piirrekombinaatioita usealla eri mallilla


    DATAN MANIPULOINTI
getFeatureMatrix palauttaa jokaista piirrematriisin 
getFluency palauttaa vektorin yhdistettyjä arvioita
getMultipleF palauttaa matriisin erillisiä arvioita
