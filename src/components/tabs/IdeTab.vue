<template>
  <div class="editor">
    <div id="ace"/>
    <div class="confirm-button">
      <UIButton text="Close" @click="close"/>
      <UIButton text="Save Changes" :primary="true" @click="save"/>
    </div>
  </div>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';
import Tabs from '@/components/tabs/Tabs.vue';
import Tab from '@/components/tabs/Tab.vue';
import Ace from 'brace';
import 'brace/mode/python';
import '@/assets/ivann-theme';
import { Getter, Mutation } from 'vuex-class';
import UIButton from '@/components/buttons/UIButton.vue';
import parse from '@/app/parser/parser';
import ParsedFunction from '@/app/parser/ParsedFunction';
import { Result } from '@/app/util';
import { ParsedFile } from '@/store/codeVault/types';
import { FuncDiff, funcsDiff } from '@/store/ManageCodevault';

@Component({
  components: {
    UIButton,
    Tab,
    Tabs,
  },
})
export default class IdeTab extends Vue {
  @Prop({ required: true }) readonly filename!: string;

  @Getter('file') file!: (filename: string) => ParsedFile;
  @Getter('usedNodes') usedNodes!:
    (diff: FuncDiff) => { name: string; deleted: string[]; changed: string[] }[];
  @Mutation('setFile') setFile!: (file: ParsedFile) => void;
  @Mutation('closeFile') closeFile!: (filename: string) => void;
  @Mutation('leaveCodeVault') leaveCodeVault!: () => void;
  @Mutation('editNodes') editNodes!: (diff: FuncDiff) => void;

  private editor?: Ace.Editor;
  private parsedFile?: Result<ParsedFunction[]>;

  mounted() {
    this.editor = Ace.edit('ace');
    this.editor.getSession().setMode('ace/mode/python');
    this.editor.setTheme('ace/theme/ivann');
    this.editor.resize(true);
    this.editor.setOptions({
      tabSize: 2,
      useSoftTabs: true,
    });
    this.editor.$blockScrolling = Infinity; // Get rid unnecessary Console info
    this.editor.on('change', this.onEditorChange);

    // Initialise file with current functions that it contains
    this.editor.setValue(this.file(this.filename).functions.join('\n'));
  }

  private save() {
    if (this.editor) {
      if (!(this.parsedFile instanceof Error) && this.parsedFile) {
        const oldFuncs = this.file(this.filename).functions;
        const newFuncs = this.parsedFile as ParsedFunction[];

        // Compare old file and new file, finding differences
        const diff: FuncDiff = funcsDiff(oldFuncs, newFuncs);

        const used = this.usedNodes(diff);

        // Warn user of effect of edit if causes change to editors
        if (diff.deleted.length > 0 || diff.changed.length > 0) {
          let warning = 'Are you sure you want to edit this file? All unsaved changes will be lost.';
          if (used.length > 0) warning = warning.concat(`\n\nWe found ${used.length} editors using this file's functions:`);
          for (const use of used) {
            warning = warning.concat(`\nIn editor "${use.name}"`);
            if (use.deleted.length > 0) warning = warning.concat(` - [${use.deleted}] will be deleted`);
            if (use.changed.length > 0) warning = warning.concat(` - [${use.changed}] will be modified`);
          }

          // STOP if user cancels
          if (!window.confirm(warning)) return;
        }

        // Run through editors using function that have been changed and update corresponding nodes
        this.editNodes(diff);

        // Override file in codevault and save
        const file = { filename: this.filename, functions: newFuncs, open: false };
        this.setFile(file);
        localStorage.setItem(`unsaved-file-${this.filename}`, JSON.stringify(file));

        // Close tab and switch to 'Functions' tab
        this.closeFile(this.filename);
        this.$emit('closeTab');
      } else {
        window.alert('Cannot save file with errors.');
      }
    }
  }

  private close() {
    this.closeFile(this.filename);
    this.$emit('closeTab');
  }

  /**
   * On Editor Change, parses the code and shows an Error if there is one.
   */
  private onEditorChange() {
    if (this.editor) {
      const code = this.editor.getValue();
      const functionsOrError = parse(code, this.filename);
      if (!(functionsOrError instanceof Error)) {
        this.editor.getSession().clearAnnotations();
      } else {
        this.showError(functionsOrError);
      }
      this.parsedFile = functionsOrError;
    }
  }

  private showError(error: Error) {
    if (this.editor) {
      this.editor.getSession().setAnnotations([{
        row: 0,
        column: 0,
        text: error.message,
        type: 'error',
      }]);
    }
  }
}
</script>

<style scoped>
.button {
  margin-right: 1rem;
}

.editor {
  margin-bottom: 1em;
  height: calc(100% - 2.5em);
}

#ace {
  border: solid var(--grey);
  border-width: 1px 0;
  height: calc(100%);
  font-size: 1em;
  font-family: monospace;
  font-weight: lighter;
}

.confirm-button {
  display: flex;
  float: right;
  padding-top: 0.5em;
  padding-bottom: 0.5em;
}
</style>
