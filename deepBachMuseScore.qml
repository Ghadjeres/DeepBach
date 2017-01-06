import QtQuick 2.1
import QtQuick.Dialogs 1.0
import QtQuick.Controls 1.0
import MuseScore 1.0
import FileIO 1.0

MuseScore {
    id: mainMuseScoreObj
    menuPath: "Plugins.deepBachMuseScore"
    description: qsTr("This plugin calls deepBach project.")
    pluginType: "dock"
    dockArea:   "left"
    FileIO {
        id: myFile
        source: tempPath() + "/" + mainMuseScoreObj.curScore.title + "_recomposed_by_deepBach.xml"
        onError: console.log(msg)
    }
    Rectangle {
        id: wrapperPanel
        color: "white"
        anchors.fill: parent
        Text {
            id: title
            text: "Deep Bach"
            font.family: "Helvetica"
            font.pointSize: 24
            color: "green"
            anchors.top: wrapperPanel.top
            anchors.topMargin: 10
            font.underline: true
        }
        Label {
            id: serverInputLabel
            wrapMode: Text.WordWrap
            text: 'Server address'
            font.pointSize:12
            anchors.left: wrapperPanel.left
            anchors.top: title.top
            anchors.topMargin: 35
        }
        TextInput {
            id: serverAddressInput
            text: "http://localhost:5000/"
            cursorVisible: true
            anchors.left: wrapperPanel.left
            anchors.top: serverInputLabel.top
            anchors.topMargin: 15
        }
        Button {
            id : buttonOpenFile
            text: qsTr("Compose")
            anchors.left: wrapperPanel.left
            anchors.top: serverAddressInput.top
            anchors.topMargin: 30
            anchors.bottomMargin: 10
            onClicked: {
                var cursor = curScore.newCursor();
                cursor.rewind(1);
                var startTick = cursor.tick;
                cursor.rewind(2);
                var endTick = cursor.tick;
                var extension = 'xml'
                // var targetFile = tempPath() + "/my_file." + extension
                myFile.remove();
                var res = writeScore(mainMuseScoreObj.curScore, myFile.source, extension)
                var content = "start_tick=" + startTick + "&end_tick=" + endTick + "&xml_string=" + encodeURIComponent(myFile.read())
                var request = new XMLHttpRequest()
                request.onreadystatechange = function() {
                    statusLabel.text = statusLabel.text + '.'
                    if (request.readyState == XMLHttpRequest.DONE) {
                        var response = request.responseText
                        // console.log("responseText : " + response)
                        if (response) {
                            statusLabel.text = 'Done'
                            myFile.write(response)
                            readScore(myFile.source)
                        } else {
                            statusLabel.text = 'Empty Response'
                        }
                    }
                }
                request.open("POST", serverAddressInput.text, true)
                request.setRequestHeader("Content-Type", "application/x-www-form-urlencoded")
                statusLabel.text = 'Loading...'
                request.send(content)
            }
        }
        Label {
            id: statusLabel
            wrapMode: Text.WordWrap
            text: ''
            color: 'green'
            font.pointSize:12
            anchors.left: wrapperPanel.left
            anchors.top: buttonOpenFile.top
            anchors.leftMargin: 10
            anchors.topMargin: 30
        }
    }
}
