<template>
  <div class="titlebar row py-2">
    <div class="col text-left">
      <img class="img-fluid titlebar-logo mr-2" src="@/assets/images/nn_logo.png" alt="IVANN"/>
      <span class="text">IVANN</span>
    </div>
    <div class="col text-center">
      <span class="text">
        MNIST-Demo
      </span>
    </div>
    <div class="col text-right">
      <span class="icon-button" @click="share">
        <i class="titlebar-icon fas fa-share-alt fa-lg mx-2"/>
      </span>
      <input
        type="file"
        id="upload-file"
        style="display: none"
        @change="load"
      >
      <span class="icon-button" @click="uploadFile">
        <i class="titlebar-icon fas fa-folder-open fa-lg mx-2"/>
      </span>
      <span class="icon-button" @click="save">
        <i class="titlebar-icon fas fa-save fa-lg mx-2"/>
      </span>
      <span class="icon-button" @click="newProject">
        <i class="titlebar-icon fas fa-file fa-lg mx-2"/>
      </span>
    </div>
  </div>
</template>

<script lang="ts">
import generateCode from '@/app/codegen/codeGenerator';
import { Component, Vue } from 'vue-property-decorator';
import { Getter, Mutation } from 'vuex-class';
import { EditorModels } from '@/store/editors/types';
import { download } from '@/file/Utils';
import { FILENAME, saveEditor, saveEditors } from '@/file/EditorAsJson';

@Component
export default class Titlebar extends Vue {
  @Getter('allEditorModels') editorModels!: EditorModels;
  @Mutation('loadEditors') loadEditors!: (file: any) => void;

  private share() {
    console.log(`Share button pressed. ${this.$data}`);
  }

  private uploadFile = () => {
    const element = document.getElementById('upload-file');
    if (!element) return;
    element.click();
  }

  private load() {
    const { files } = document.getElementById('upload-file') as HTMLInputElement;
    if (!files) return;

    const fr: FileReader = new FileReader();
    fr.onload = (event) => {
      if (!event.target) return;

      // Load all editors using parsed file
      this.loadEditors(JSON.parse(event.target.result as string));
    };

    // Trigger the file to be read
    fr.readAsText(files[0]);
  }

  private save() {
    const {
      overviewEditor,
      modelEditors,
      dataEditors,
      trainEditors,
    } = this.editorModels;

    const editorsSaved = {
      overviewEditor: saveEditor(overviewEditor),
      modelEditors: saveEditors(modelEditors),
      dataEditors: saveEditors(dataEditors),
      trainEditors: saveEditors(trainEditors),
    };

    download(FILENAME, JSON.stringify(editorsSaved));
  }

  private newProject() {
    // TODO FE-55 Implement better Confirm dialog
    if (window.confirm('Are you sure you want to create a new project? '
      + 'All unsaved progress will be lost.')) {
      const blankProject = {
        overviewEditor: {
          name: 'Overview',
          editorState: {
            nodes: [],
            connections: [],
          },
        },
        modelEditors: [{
          name: 'untitled',
          editorState: {
            nodes: [],
            connections: [],
            panning: {
              x: 0,
              y: 0,
            },
            scaling: 1,
          },
        }],
        dataEditors: [],
        trainEditors: [],
      };
      this.loadEditors(blankProject);
      this.$cookies.remove('unsaved');
    }
  }
}
</script>

<style lang="scss" scoped>
  .titlebar {
    height: 2.5rem;
    background-color: var(--background-alt);

    border-bottom: 0.08rem solid var(--grey);
  }

  .titlebar-logo {
    height: 1.2rem;
  }

  .text {
    color: var(--foreground);
  }

  .titlebar-icon {
    color: var(--foreground);
  }

  .icon-button {
    background-color: var(--background-alt);
    width: 5rem;
    height: 5rem;
    margin-left: 0.15rem;
    margin-right: 0.15rem;
    padding: 0.3rem 0.1rem;
    position: relative;
    top: 0;

    &:hover {
      background-color: #2c2c2c;
      cursor: pointer;
    }
  }
</style>
