import QtQuick 2.2
import QtQuick.Dialogs 1.2
import QtQuick.Controls 1.1
import MuseScore 3.0
import FileIO 3.0

MuseScore {
    id: mainMuseScoreObj
    menuPath: "Plugins.deepBachMuseScore"
    description: qsTr("This plugin calls deepBach project.")
    pluginType: "dock"
    dockArea:   "left"
    property variant serverCalled: false
    property variant loading: false
    property variant linesLogged: 0
    property variant serverAddress: "http://localhost:5000/"
    FileIO {
        id: myFile
        source: "/tmp/deepbach.mxl"
        onError: console.log(msg)
    }
    FileIO {
        id: myFileXml
        source: "/tmp/deepbach.xml"
        onError: console.log(msg)
    }
    onRun: {
        console.log('on run called')
        if (mainMuseScoreObj.serverCalled !== serverAddress || modelSelector.model.length === 0) {
            var requestModels = getRequestObj('GET', 'models')
            if (call(
                requestModels,
                false,
                function(responseText){
                    mainMuseScoreObj.serverCalled = serverAddress;
                    try {
                        modelSelector.model = JSON.parse(responseText)
                        logStatus('Models list loaded')
                        var requestLoadModel = getRequestObj("GET", 'current_model')
                        call(
                            requestLoadModel,
                            false,
                            function(response) {
                                logStatus('currently loaded model is ' + response)
                                for (var i in modelSelector.model) {
                                    if (modelSelector.model[i] === response) {
                                        console.log('set selected at ' + i)
                                        modelSelector.currentIndex = i
                                    }
                                }
                            }
                        )
                    } catch(error) {
                        console.log(error)
                        logStatus('No models found')
                    }

                }
            )) {
                logStatus('Retrieving models list at ' + serverAddress)
            }
        }
    }
    Rectangle {
        id: wrapperPanel
        color: "white"
        Text {
            id: title
            text: "DeepBach"
            font.family: "Helvetica"
            font.pointSize: 20
            color: "black"
            anchors.top: wrapperPanel.top
            anchors.topMargin: 10
            font.underline: false
        }
        Button {
            id : loadModel
            anchors.top: title.bottom 
            anchors.topMargin: 15
            text: qsTr("Load")
            onClicked: {
                if (modelSelector.model[modelSelector.currentIndex]) {
                    var requestLoadModel = getRequestObj("POST", 'current_model')
                    if (call(
                        requestLoadModel,
                        {
                            model_name: modelSelector.model[modelSelector.currentIndex]
                        }, 
                        function(response) {
                            logStatus(response)
                        }
                    )) {
                        logStatus('Loading model ' + modelSelector.model[modelSelector.currentIndex])
                    }
                }
            }
        }
        ComboBox {
            id: modelSelector
            anchors.top: modelSelector.top
            anchors.left: modelSelector.right
            anchors.leftMargin: 10
            model: []
            width: 100
            visible: false
        }
        Button {
            id : buttonOpenFile
            text: qsTr("Compose")
            anchors.top: loadModel.top
            anchors.topMargin: 30
            anchors.bottomMargin: 10
            onClicked: {
                var cursor = curScore.newCursor();
                cursor.rewind(1);
                var startStaff = cursor.staffIdx;
                var startTick = cursor.tick;
                cursor.rewind(2);
                var endStaff = cursor.staffIdx;
                var endTick = cursor.tick;
                var extension = 'mxl'
                // var targetFile = tempPath() + "/my_file." + extension
                myFile.remove();
                logStatus(myFile.source);
                var res = writeScore(mainMuseScoreObj.curScore, myFile.source, extension)
                logStatus(res);
                var request = getRequestObj("POST", 'compose')
                if (call(
                    request,
                    {
                        start_staff: startStaff,
                        end_staff: endStaff,
                        start_tick: startTick,
                        end_tick: endTick,
//                        xml_string: myFile.read(),
                        file_path: myFile.source
                    },
                    function(response) {
                        if (response) {
                            logStatus('Done composing')
                            // myFileXml.write(response)
                            readScore(myFileXml.source)		

                        } else {
                            logStatus('Got Empty Response when composing')
                        }
                    }
                )) {
                    logStatus('Composing...')
                }
            }
        }
        Label {
            id: statusLabel
            wrapMode: Text.WordWrap
            text: ''
            color: 'grey'
            font.pointSize:12
            anchors.left: wrapperPanel.left
            anchors.top: buttonOpenFile.top
            anchors.leftMargin: 10
            anchors.topMargin: 30
            visible: false
        }
    }
    function logStatus(text) {
        mainMuseScoreObj.linesLogged++;
        if (mainMuseScoreObj.linesLogged > 15) {
            // break the textblock into an array of lines
            var lines = statusLabel.text.split("\r\n");
            // remove one line, starting at the first position
            lines.splice(0,1);
            // join the array back into a single string
            statusLabel.text = lines.join("\r\n");
        }
        statusLabel.text += '- ' + text + "\r\n"
    }
    function getRequestObj(method, endpoint) {
        console.debug('calling endpoint ' + endpoint)
        var request = new XMLHttpRequest()
        endpoint = endpoint || ''
        request.open(method, serverAddress + endpoint, true)
        return request
    }
    function call(request, params, cb) {
        if (mainMuseScoreObj.loading) {
            logStatus('refusing to call server')
            return false
        }
        request.onreadystatechange = function() {
            if (request.readyState == XMLHttpRequest.DONE) {
                mainMuseScoreObj.loading = false;
                cb(request.responseText);
            }
        }
        if (params) {
            request.setRequestHeader("Content-Type", "application/x-www-form-urlencoded")
            var pairs = [];
            for (var prop in params) {
              if (params.hasOwnProperty(prop)) {
                var k = encodeURIComponent(prop),
                    v = encodeURIComponent(params[prop]);
                pairs.push( k + "=" + v);
              }
            }

            const content = pairs.join('&')
            console.debug('params ' + content)
            mainMuseScoreObj.loading = true;
            request.send(content)
        } else {
            mainMuseScoreObj.loading = true;
            request.send()
        }
        return true
    }
}
