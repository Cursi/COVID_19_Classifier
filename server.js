const express = require('express')
const app = express()
const port = process.env.PORT || 3000

const bodyParser = require('body-parser');
app.use(bodyParser.json({ limit: '50mb' }));

const PS = require('python-shell');

app.get('/', (req, res) => 
{
    res.sendFile(__dirname + "/index.html");
})

var pyOutput = null;

app.post("/upload", (req, res) =>
{
    if(req.body.message)
    {
        if(pyOutput != null)
        {
            let response = { status: 200, output: pyOutput};

            if(pyOutput.includes("PROCESSING_ERROR"))
                response.status = 400;

            res.write(JSON.stringify(response));
            pyOutput = null;
            res.end();
        }
        else
            res.end();
    }
    // TODO: De trimis numele fisierului ca parametru
    else if(req.body.fileName)
    {
        console.log(req.body.fileName);

        var pyshell = new PS.PythonShell(`${__dirname}/ML/classifier.py`);
        pyshell.send(req.body.content);

        let waitResponse = { status: 200, output: "PROCESSING"}
        res.write(JSON.stringify(waitResponse));
        res.end();

        pyshell.stdout.on('data', function(data) 
        {
            pyOutput = data;
        });
    }
});

app.listen(port, () => 
{
    console.log(`Example app listening on ${port}`);
})