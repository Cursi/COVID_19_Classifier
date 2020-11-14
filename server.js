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

app.post("/upload", (req, res) =>
{
    console.log(req.body.fileName);

    // var pyshell = new PS.PythonShell('cursi.py');
    var pyshell = new PS.PythonShell('encoding.py');
    pyshell.send(req.body.content);

    pyshell.stdout.on('data', function(data) 
    {
        let response = { status: 200, output: data};

        if(data.includes("PROCESSING_ERROR"))
            response.status = 400;

        res.write(JSON.stringify(response));
        res.end();
    });

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