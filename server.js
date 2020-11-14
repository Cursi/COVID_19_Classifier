const express = require('express')
const app = express()
const port = process.env.PORT || 3000

var cmd = require("node-cmd");

app.get('/', (req, res) => 
{
    const pyProg = cmd.run("python dummy.py").stdout.on('data', function(data) 
    {
        console.log(data.toString());
        res.write(data);
        res.end();
    });
})

app.listen(port, () => 
{
    console.log(`Example app listening on ${port}`)
})