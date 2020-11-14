const express = require('express')
const app = express()
const port = process.env.PORT || 3000

// const fileUpload = require('express-fileupload');
// app.use(fileUpload());

const bodyParser = require('body-parser');
app.use(bodyParser.json({ limit: '50mb' }));

const pyCmd = require("node-cmd");
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
    else if(req.body.fileName)
    {
        console.log(req.body.fileName);

        // var pyshell = new PS.PythonShell('cursi.py');
        var pyshell = new PS.PythonShell('encoding.py');
        pyshell.send(req.body.content);

        let waitResponse = { status: 200, output: "PROCESSING"}
        res.write(JSON.stringify(waitResponse));
        res.end();

        pyshell.stdout.on('data', function(data) 
        {
            pyOutput = data;
        });
    }

    // console.log(req.files);

    // if(req.files)
    // {
    //     let file = req.files.dataset;
    //     let filename = file.name;

    //     // console.log(Buffer.from(file.data).toString("base64"));

    //     var pyshell = new PS.PythonShell('cursi.py');
    //     pyshell.send(Buffer.from(file.data).toString("base64"));

    //     pyshell.stdout.on('data', function(data) 
    //     {
    //         console.log(data.toString());
    //         res.write(data);
    //         res.end();
    //     });

        // file.mv(`${__dirname}/upload/${filename}`, (err) =>
        // {
        //     if(err)
        //     {
        //         console.log(err);
        //         res.send("Eroare boss");
        //     }
        //     else
        //     {
        //         pyCmd.run(`python dummy.py ${__dirname}/upload/${filename}`).stdout.on('data', function(data) 
        //         {
        //             console.log(data.toString());
        //             res.write(data);
        //             res.end();
        //         });
        //     }
        // });
    // }
});

app.listen(port, () => 
{
    console.log(`Example app listening on ${port}`);
})