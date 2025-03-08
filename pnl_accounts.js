const accountModule = (function() {
    const accountDescriptions = {
        "1000": "Cassa ",
        "1001": "Cassa Euro",
        "1020": "Conto Raiffeisen CHF - linea di credito",
        "1021": "Conto Raiffeisen EUR",
        "1023": "Conto Raiffeisen CHF",
        "1025": "Conto corrente postale",
        "1026": "Conto corrente postale - riserve",
        "1027": "Paypal CHF",
        "1090": "Conto di trasferimento",
        "1095": "cc Di Gregorio",
        "1100": "Clienti",
        "1109": "Delcredere",
        "1300": "Costi anticipati",
        "1301": "Ricavi da incassare",
        "1510": "Mobilio e installazioni",
        "1520": "Macchine ufficio, informatica e comunicazione",
        "1526": "Software",
        "2000": "Creditori",
        "2050": "cc AVS",
        "2051": "cc Cassa Pensione LPP",
        "2052": "cc malattia",
        "2053": "cc Zurich infortuni",
        "2054": "cc Imposte alla fonte",
        "2055": "cc Indennità maternità",
        "2056": "cc Cassa assegni familiari",
        "2062": "cc Stipendi altri",
        "2070": "cc Carta di credito",
        "2099": "cc Servizio di fatturazione",
        "2261": "Dividendi",
        "2281": "Imposta preventiva su dividendi",
        "2600": "Accantonamenti",
        "2730": "Transitori passivi",
        "2800": "Capitale proprio / capitale sociale",
        "2900": "Riserva legale da capitale",
        "2990": "Utile o perdita riportata",
        "3400": "Ricavi da corsi di formazione",
        "3401": "Ricavi da servizio di fatturazione",
        "3805": "Perdite su debitori",
        "6850": "Interessi Attivi",
        "8510": "Ricavi straordinari",
        "4400": "Costi per prestazioni di terzi",
        "4401": "Costi per consulenti interni",
        "4402": "Costi ausiliari per prestazioni di terzi",
        "4430": "Dispense",
        "4460": "Costi associativi",
        "5000": "Salari",
        "5001": "Premi prestazioni",
        "5700": "Contributi AVS/AI/IPG",
        "5720": "Contributi previdenza professionale",
        "5730": "Assicurazione infortuni",
        "5740": "Premi assicurazione malattia",
        "5750": "Indennità maternità",
        "5760": "Indennità COVID19",
        "5790": "Imposta alla fonte",
        "5820": "Spese di viaggio",
        "5830": "Rimborsi spese forfettarie",
        "5870": "Costi assicurativi",
        "5880": "Altre spese personale (costi senza cliente)",
        "5890": "Costi formazione ed aggiornamento",
        "6000": "Pigione/affitto aule",
        "6105": "Costi leasing immobilizzazioni mobiliari",
        "6400": "Costi energia",
        "6500": "Spese ufficio/amministrazione",
        "6550": "Spese di consulenza amministrativa",
        "6551": "Spese telefoniche/internet",
        "6552": "Spese postali",
        "6600": "Costi di pubblicità e marketing",
        "6650": "Spese di rappresentanza",
        "6660": "Costi software",
        "6700": "Altri costi",
        "6800": "Interessi e spese bancarie",
        "6842": "Differenze di cambio/Arrotondamenti",
        "6900": "Ammortamenti",
        "8500": "Costi straordinari",
        "8900": "Imposte"
    };

    function getAccountDescription(accountCode) {
        return accountDescriptions[accountCode] || "n/a";
    }

    function getAllAccountKeys() {
        return Object.keys(accountDescriptions);
    }

    return {
        getAccountDescription,
        getAllAccountKeys
    };
})();

// Esportazione per uso in altri file
if (typeof module !== 'undefined') {
    module.exports = accountModule;
}
