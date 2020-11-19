<template>
  <div class="h-100">
    <div id="ace"/>
    <div class="save-banner">
      <div>
        <UIButton text="Cancel" @click="cancel"/>
      </div>
      <div>
        <UIButton text="Save Changes" :primary="true" @click="save"/>
      </div>
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
import { Mutation } from 'vuex-class';
import Custom from '@/nodes/common/Custom';
import UIButton from '@/components/buttons/UIButton.vue';
import parse from '@/app/parser/parser';
import ParsedFunction from '@/app/parser/ParsedFunction';
import { Result } from '@/app/util';
import { ParsedFile } from '@/store/codeVault/types';

@Component({
  components: {
    UIButton,
    Tab,
    Tabs,
  },
})
export default class IdeTab extends Vue {
  @Prop({ required: true }) readonly filename!: string;

  @Mutation('setFile') setFile!: (file: ParsedFile) => void;
  @Mutation('closeFile') closeFile!: (filename: string) => void;
  @Mutation('leaveCodeVault') leaveCodeVault!: () => void;
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
  }

  private save() {
    if (this.editor) {
      if (!(this.parsedFile instanceof Error) && this.parsedFile) {
        const file = { filename: this.filename, functions: this.parsedFile, open: false };
        this.setFile(file);
        this.closeFile(this.filename);
        this.$cookies.set(`unsaved-file-${this.filename}`, file);
        this.$emit('closeTab');
      } else {
        window.alert('Cannot save file with errors.');
      }
    }
  }

  private cancel() {
    this.closeFile(this.filename);
    this.$emit('closeTab');
  }

  /**
   * On Editor Change, parses the code and shows an Error if there is one.
   */
  private onEditorChange() {
    if (this.editor) {
      const code = this.editor.getValue();
      const functionsOrError = parse(code);
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
.save-banner {
  border-top: 1px solid var(--grey);
  background: var(--background);
  height: 3em;
  display: flex;
  padding-left: calc(100% - 15.5em);
}
.button {
  margin-top: 14px;
  margin-right: 1rem;
}
#ace {
  border-top: 1px solid var(--grey);
  height: calc(100% - 3em);
  font-size: 1em;
  font-family: monospace;
  font-weight: lighter;
}
</style>
