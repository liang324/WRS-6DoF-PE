function updateStatus(data) {
    document.getElementById("status").innerText = JSON.stringify(data, null, 2);
}

function sendInit() {
    fetch("/robot/init", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({})
    })
    .then(res => res.json())
    .then(data => updateStatus(data))
    .catch(err => updateStatus({"status": "error", "msg": err}));
}

function sendCmd(cmd) {
    fetch("/robot/cmd", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({cmd: cmd})
    })
    .then(res => res.json())
    .then(data => updateStatus(data))
    .catch(err => updateStatus({"status": "error", "msg": err}));
}
