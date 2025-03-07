def total_euro(radni_sati, satnica):
    return radni_sati * satnica

if __name__ == "__main__":
        radni_sati = float(input("Radni sati: "))
        satnica = float(input("eura/h: "))

        ukupno = total_euro(radni_sati, satnica)

        print(f"Ukupno: {ukupno} eura")