# Što se događa s procesom učenja:
# 1. ako se koristi jako velika ili jako mala veličina serije? 
    # Prevelika veličina serije može dovesti do slabijeg učenja,
    # treba više memorije i može uzrokovati probleme s konvergencijom.
    # Premala veličina serije može dovesti do vrlo nestabilnog učenja,
    # te rezultati jako variraju
# 2. ako koristite jako malu ili jako veliku vrijednost stope učenja?
    # Prevelika vrijednost stope učenja dovodi do jake oscilacije te 
    # dovodi do velikih gubitaka i loših rezlultata
    # Premala vrijednost atope učenja dovodi do sporog učenja i 
    # treba puno epoha da bi se postigla konvergencija
# 3. ako izbacite određene slojeve iz mreže kako biste dobili manju mrežu?
    # Manja mreža ima manje parametara, brže uči i manje koristi memoriju, 
    # ali može imati slabije rezultate tj točnost zbog nedostataka slojeva
# 4. ako za 50% smanjite veličinu skupa za učenje?
    # Ako smanjimo veličinu skupa za učenja za 50% model će imati manje
    # podataka za učenje, istodobno se povećava rizik od ovrefittinga
    # te će rezultati na testom skupu biti lošiji