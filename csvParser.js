let lastLoadedFile = null;

const configs = {
    postfinance: ";\n8;3\nD:dd.mm.yyyy;T;A;A;X;X",
    raiffeisen: ";\n1;0\nX;D:yyyy-mm-dd;T;A;X;X"
};

document.querySelectorAll('input[name="bank"]').forEach(radio => {
    radio.addEventListener('change', function() {
        document.getElementById('config').value = configs[this.id];
    });
});

document.getElementById('fileInput').addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (!file) return;
    lastLoadedFile = file;
    loadCSV(file);
});

document.getElementById('reloadBtn').addEventListener('click', function() {
    if (lastLoadedFile) {
        loadCSV(lastLoadedFile);
    }
});

function loadCSV(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        const text = e.target.result;
        processCSV(text);
    };
    reader.readAsText(file);
}

function parseDate(dateStr, format) {
    const parts = dateStr.match(/\d+/g);
    if (!parts) return dateStr;

    const yIndex = format.indexOf('yyyy');
    const mIndex = format.indexOf('mm');
    const dIndex = format.indexOf('dd');

    let year = yIndex !== -1 ? parts[format.split(/[^ymd]/).indexOf('yyyy')] : '';
    let month = mIndex !== -1 ? parts[format.split(/[^ymd]/).indexOf('mm')].padStart(2, '0') : '';
    let day = dIndex !== -1 ? parts[format.split(/[^ymd]/).indexOf('dd')].padStart(2, '0') : '';

    return `${day}.${month}.${year}`;
}

function processCSV(text) {
    const config = document.getElementById('config').value.split('\n');
    const separator = config[0];
    const [skipStart, skipEnd] = config[1].split(separator).map(Number);
    const mask = config[2].split(separator);

    const rows = text.trim().split('\n');
    const dataRows = rows.slice(skipStart, skipEnd ? -skipEnd : undefined);
    
    const tableBody = document.querySelector('#dataTable tbody');
    tableBody.innerHTML = '';
    
    dataRows.forEach(row => {
        const columns = row.split(separator);
        let date = '', desc = '', amount = 0, dateFormat = '';
        
        mask.forEach((m, index) => {
            if (m.startsWith('D:')) {
                dateFormat = m.split(':')[1];
                date = parseDate(columns[index], dateFormat);
            } else if (m === 'T') desc = columns[index].replace(/"/g, "");
            else if (m === 'A') amount += parseFloat(columns[index]) || 0;
            else if (m === 'A-') amount -= parseFloat(columns[index]) || 0;
        });
        
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${date}</td><td>${desc}</td><td>${amount.toFixed(2)}</td>`;
        tableBody.appendChild(tr);
    });
}
